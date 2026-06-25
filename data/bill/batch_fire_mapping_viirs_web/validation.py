"""Typed bounds for subprocess arguments and user-supplied fire creation
inputs. Pure functions, no state."""

import datetime
import math

# VNP14IMG (the LAADS active-fire product) became available 2012-01-19.
# Earlier dates have no satellite passes and the LAADS API will return
# empty results, so reject up front with an actionable message.
VIIRS_VNP14IMG_FIRST_AVAILABLE = datetime.date(2012, 1, 19)


_PARAM_SPEC = {
    'seed':                 ('int',   0, 2**31 - 1),
    'rf_n_estimators':      ('int',   1, 2000),
    'rf_max_depth':         ('int',   1, 100),
    'rf_max_features':      ('choice', {'sqrt', 'log2', 'auto'}),
    'rf_random_state':      ('int',   0, 2**31 - 1),
    'controlled_ratio':     ('float', 0.01, 2.0),
    'hdbscan_min_samples':  ('int',   1, 10000),
    'tsne_perplexity':      ('float', 1.0, 1000.0),
    'tsne_learning_rate':   ('float', 1.0, 10000.0),
    'tsne_max_iter':        ('int',   100, 10000),
    'tsne_init':            ('choice', {'random', 'pca'}),
    'tsne_n_components':    ('int',   2, 3),
    'tsne_random_state':    ('int',   0, 2**31 - 1),
    'contour_width':        ('float', 0.0, 10.0),
    'sample_rate':          ('float', 0.001, 1.0),
    'min_samples':          ('int',   1, 1000000),
    'max_samples':          ('int',   1, 1000000),
    'brush_size':           ('int',   1, 10000),
    'point_threshold':      ('int',   1, 10_000_000),
    'brush_all_segments':   ('bool',),
    # A12 hint-aware brush — controls per-component scoring.
    'hint_aware_brush':       ('bool',),
    'brush_score_threshold':  ('float', 0.0, 1.0),
    'brush_proximity_frac':   ('float', 0.0, 1.0),
    # B3 — keep class_brush scratch files (debug-only).
    'brush_keep_intermediates': ('bool',),
    # A1 — stratified sampling.
    'stratify':                ('bool',),
    'stratify_inside_ratio':   ('float', 0.05, 0.95),
    # A4 — robust per-band z-scoring.
    'scale_features':          ('bool',),
    # A8 — spatial coherence weight.
    'spatial_weight':          ('float', 0.0, 5.0),
    # A5 — cluster-vote score threshold.
    'cluster_score_threshold': ('float', 0.0, 1.0),
}


def _validate_param(key: str, raw):
    """Validate and coerce a single parameter. Raises ValueError on bad input."""
    if key not in _PARAM_SPEC:
        return raw
    spec = _PARAM_SPEC[key]
    kind = spec[0]
    if kind == 'int':
        _, lo, hi = spec
        v = int(float(raw))
        if not (lo <= v <= hi):
            raise ValueError(f'{key}={raw} out of range [{lo}, {hi}]')
        return v
    if kind == 'float':
        _, lo, hi = spec
        v = float(raw)
        if not (lo <= v <= hi):
            raise ValueError(f'{key}={raw} out of range [{lo}, {hi}]')
        return v
    if kind == 'choice':
        _, allowed = spec
        s = str(raw)
        if s not in allowed:
            raise ValueError(
                f'{key}={raw} must be one of {sorted(allowed)}')
        return s
    if kind == 'bool':
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in ('1', 'true', 'yes', 'on'):
            return True
        if s in ('0', 'false', 'no', 'off', ''):
            return False
        raise ValueError(f'{key}={raw} must be boolean')
    return raw


def _validate_embed_bands(eb) -> str | None:
    """Validate embed_bands (comma-separated positive ints). Returns cleaned string."""
    if not eb or not str(eb).strip():
        return None
    parts = [p.strip() for p in str(eb).split(',') if p.strip()]
    if not all(p.isdigit() and 1 <= int(p) <= 999 for p in parts):
        raise ValueError(f'Invalid embed_bands: {eb!r}')
    return ','.join(parts)


# ---------------------------------------------------------------------------
# User-supplied fire-creation input validators
# ---------------------------------------------------------------------------

def _validate_fire_name(name, existing_names=()):
    """Validate a user-supplied fire name. Returns the canonical name
    (unchanged on success) or raises ValueError.

    Rules: 1-64 chars, alnum first char, [A-Za-z0-9_. -] thereafter,
    no '..', no '/' or '\\'. Uniqueness is checked case-insensitively
    against ``existing_names`` (any iterable of strings).
    """
    from .state import _is_valid_fire_name as _ok
    if not isinstance(name, str):
        raise ValueError('Fire name must be a string')
    name = name.strip()
    if not name:
        raise ValueError('Fire name is required')
    if not _ok(name):
        raise ValueError(
            'Fire name must start with a letter or digit and contain only '
            'letters, digits, spaces, underscores, dots, or hyphens '
            '(max 64 chars). Path separators and ".." are not allowed.')
    name_lc = name.lower()
    for existing in existing_names:
        if str(existing).lower() == name_lc:
            raise ValueError(
                f'Fire name "{name}" is already in use (case-insensitive)')
    return name


def _validate_date(value, field_name='date'):
    """Parse a YYYY-MM-DD string into a ``datetime.date``. Raises
    ValueError on bad input. Empty/None propagates as None — callers
    must handle defaults before passing here when emptiness is allowed.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise ValueError(f'{field_name} must be a YYYY-MM-DD string')
    s = value.strip()
    # Strict: only YYYY-MM-DD; reject slashes / extra time components.
    if len(s) != 10 or s[4] != '-' or s[7] != '-':
        raise ValueError(
            f'{field_name}="{value}" must be in YYYY-MM-DD format')
    try:
        d = datetime.date.fromisoformat(s)
    except ValueError as exc:
        raise ValueError(
            f'{field_name}="{value}" is not a valid date: {exc}')
    return d


def _validate_date_range(start_str, end_str,
                         default_start='', default_end='',
                         today=None):
    """Validate a (start, end) date pair for VIIRS download.

    Empty strings fall through to the supplied defaults (``default_start``
    / ``default_end``). Returns ``(start_date, end_date)`` as
    ``datetime.date`` objects. Raises ValueError on any failure.

    Constraints:
      - both parseable as YYYY-MM-DD
      - start <= end
      - start >= 2012-01-19 (VNP14IMG availability)
      - end <= today (server-time)
    """
    today = today or datetime.date.today()
    s_raw = start_str if (start_str and str(start_str).strip()) \
        else default_start
    e_raw = end_str if (end_str and str(end_str).strip()) else default_end
    start = _validate_date(s_raw, 'start')
    end = _validate_date(e_raw, 'end')
    if start is None:
        raise ValueError('start date is required')
    if end is None:
        raise ValueError('end date is required')
    if start > end:
        raise ValueError(
            f'start date {start} must be on or before end date {end}')
    if start < VIIRS_VNP14IMG_FIRST_AVAILABLE:
        raise ValueError(
            f'start date {start} is before VIIRS VNP14IMG availability '
            f'({VIIRS_VNP14IMG_FIRST_AVAILABLE})')
    if end > today:
        raise ValueError(
            f'end date {end} is in the future (today is {today})')
    return start, end


def _validate_bbox(bbox, raster_extent):
    """Validate a user-drawn bbox against the year's raster extent.

    ``bbox``: 4-element sequence (xmin, ymin, xmax, ymax) in raster CRS.
    ``raster_extent``: 4-tuple (rxmin, rymin, rxmax, rymax) for clipping.

    Returns the (clipped) bbox as a tuple of four floats. Raises
    ValueError on:
      - wrong arity
      - non-finite values
      - zero/negative width or height after clipping
      - no overlap with raster
    """
    if bbox is None or not hasattr(bbox, '__len__') or len(bbox) != 4:
        raise ValueError('bbox must be 4 numbers [xmin, ymin, xmax, ymax]')
    try:
        xmin, ymin, xmax, ymax = (float(v) for v in bbox)
    except (TypeError, ValueError):
        raise ValueError('bbox values must be numbers')
    for label, v in (('xmin', xmin), ('ymin', ymin),
                      ('xmax', xmax), ('ymax', ymax)):
        if not math.isfinite(v):
            raise ValueError(f'bbox {label}={v} is not finite')
    if xmin >= xmax:
        raise ValueError(f'bbox xmin ({xmin}) must be < xmax ({xmax})')
    if ymin >= ymax:
        raise ValueError(f'bbox ymin ({ymin}) must be < ymax ({ymax})')
    rxmin, rymin, rxmax, rymax = (float(v) for v in raster_extent)
    # No-overlap check
    if xmax <= rxmin or xmin >= rxmax \
            or ymax <= rymin or ymin >= rymax:
        raise ValueError(
            f'bbox does not overlap raster extent '
            f'[{rxmin}, {rymin}, {rxmax}, {rymax}]')
    # Clip to raster
    cxmin = max(xmin, rxmin)
    cymin = max(ymin, rymin)
    cxmax = min(xmax, rxmax)
    cymax = min(ymax, rymax)
    if cxmin >= cxmax or cymin >= cymax:
        raise ValueError(
            'bbox clipped to raster extent has zero area')
    return (cxmin, cymin, cxmax, cymax)
