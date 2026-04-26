"""Typed bounds for subprocess arguments. Pure functions, no state."""

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
