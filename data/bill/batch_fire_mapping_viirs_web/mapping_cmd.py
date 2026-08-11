"""Build the fire_mapping_cli.py subprocess command from a fire + params dict."""

import os
import sys

from .state import AppState, FireInfo
from .validation import _validate_param, _validate_embed_bands

state: AppState = None


# Bands kept by the most recent reduction, in output order (original
# 0-based indices). Consumed when embed_bands is remapped, so the two
# cannot disagree.
_kept_band_map = []


def _fire_exclusions(fire):
    """The three band exclusions for *fire*, defaults applied."""
    return (bool(getattr(fire, 'exclude_b8', True)),
            bool(getattr(fire, 'exclude_pre_fire', True)),
            bool(getattr(fire, 'exclude_diff', True)))


def reduced_stack(src_path: str, fire, log=None):
    """A copy of *src_path* holding only the bands the model should see.

    Built as a real file rather than by teaching the CLI to skip bands:
    the CLI treats its input as an opaque stack, and every downstream
    artifact (classified raster, geo sidecars, previews) keys off that
    file's geometry. Reducing here keeps all of that untouched.

    Cached beside the source, with the exclusion combination in the
    filename -- a cache keyed only on "reduced" would serve a stack
    built for different settings and silently feed the classifier the
    wrong bands.

    Returns the original path when nothing is excluded, or on any
    failure: a wider stack still produces a result, an absent one does
    not.
    """
    global _kept_band_map
    try:
        from osgeo import gdal
        from .band_select import select_bands, selection_tag

        x_b8, x_pre, x_diff = _fire_exclusions(fire)
        if not (x_b8 or x_pre or x_diff):
            _kept_band_map = []
            return src_path

        ds = gdal.Open(src_path, gdal.GA_ReadOnly)
        if ds is None:
            return src_path
        names = [ds.GetRasterBand(i + 1).GetDescription() or ''
                 for i in range(ds.RasterCount)]

        sel = select_bands(names, x_b8, x_pre, x_diff, log=log)
        keep = sel['keep']
        _kept_band_map = list(keep)
        if len(keep) == len(names):
            # Nothing actually matched (e.g. a stack with no B8).
            ds = None
            return src_path

        tag = selection_tag(x_b8, x_pre, x_diff)
        out_path = f'{os.path.splitext(src_path)[0]}_{tag}.bin'
        try:
            if (os.path.isfile(out_path)
                    and os.path.getmtime(out_path)
                    >= os.path.getmtime(src_path)):
                ds = None
                return out_path
        except OSError:
            pass

        drv = gdal.GetDriverByName('ENVI')
        out = drv.Create(out_path, ds.RasterXSize, ds.RasterYSize,
                         len(keep), gdal.GDT_Float32,
                         options=['INTERLEAVE=BSQ'])
        out.SetGeoTransform(ds.GetGeoTransform())
        out.SetProjection(ds.GetProjection())
        for j_, i_ in enumerate(keep, start=1):
            b = out.GetRasterBand(j_)
            b.WriteArray(ds.GetRasterBand(i_ + 1).ReadAsArray())
            b.SetDescription(names[i_])
            b = None
        out = None
        ds = None
        sys.stderr.write(
            f'[bands] classifier input -> {os.path.basename(out_path)}\n')
        return out_path
    except Exception as exc:
        sys.stderr.write(
            f'[bands] could not build the reduced stack ({exc}); '
            f'using the full stack\n')
        _kept_band_map = []
        return src_path


def _build_mapping_cmd(fire: FireInfo, params: dict,
                       save_state: str = None,
                       load_state: str = None) -> list[str]:
    """Build the subprocess command for fire_mapping_cli.py.

    Raises ValueError if any parameter fails validation.
    """
    rate = params.get('sample_rate')
    rate = float(rate) if rate is not None else state.sample_rate
    min_s = params.get('min_samples')
    min_s = int(min_s) if min_s is not None else state.min_samples
    max_s = params.get('max_samples')
    max_s = int(max_s) if max_s is not None else state.max_samples
    sample_size = int(round(fire.crop_w * fire.crop_h * rate))
    sample_size = max(min_s, min(max_s, sample_size))

    if not fire.hint_bin or not os.path.isfile(fire.hint_bin):
        raise ValueError(
            'No hint mask available. Switch hint mode to '
            '"Red wins (post)" or "Red wins (diff)" first, or '
            'ensure VIIRS data is available for this fire.')

    # '-u' forces line-buffered stdout on the CLI child. Without it,
    # Python block-buffers to a pipe (~8 KB), so the web UI sees many
    # stage-transition lines arrive in one burst and the progress pills
    # appear to flip to "done" all at once. With '-u' each print flushes
    # immediately and the UI can animate each stage individually.
    cmd = [
        sys.executable,
        '-u',
        state.cli_script,
        '--sample_size', str(sample_size),
        reduced_stack(fire.crop_bin, fire,
                      log=lambda m: fire.console_log.append(m)),
        fire.hint_bin,
        '--fire_numbe', fire.fire_numbe,
        '--start_date', fire.acc_start,
        '--end_date', fire.acc_end,
    ]

    if fire.perim_bin and os.path.exists(fire.perim_bin):
        cmd += ['--perimeter', fire.perim_bin]

    if save_state:
        cmd += ['--save_state', save_state]
    if load_state:
        cmd += ['--load_state', load_state]

    flag_map = {
        'seed': '--seed',
        'rf_n_estimators': '--rf_n_estimators',
        'rf_max_depth': '--rf_max_depth',
        'rf_max_features': '--rf_max_features',
        'rf_random_state': '--rf_random_state',
        'controlled_ratio': '--controlled_ratio',
        'hdbscan_min_samples': '--hdbscan_min_samples',
        'tsne_perplexity': '--tsne_perplexity',
        'tsne_learning_rate': '--tsne_learning_rate',
        'tsne_max_iter': '--tsne_max_iter',
        'tsne_init': '--tsne_init',
        'tsne_n_components': '--tsne_n_components',
        'tsne_random_state': '--tsne_random_state',
        'contour_width': '--contour_width',
        'brush_size': '--brush_size',
        'point_threshold': '--point_threshold',
        # A12 hint-aware brush thresholds — passed verbatim.
        'brush_score_threshold': '--brush_score_threshold',
        'brush_proximity_frac':  '--brush_proximity_frac',
        # A1 / A8 / A5 numeric knobs.
        'stratify_inside_ratio':  '--stratify_inside_ratio',
        'spatial_weight':         '--spatial_weight',
        'cluster_score_threshold': '--cluster_score_threshold',
    }

    for key, flag in flag_map.items():
        val = params.get(key)
        if val is not None and str(val).strip():
            val = _validate_param(key, val)
            # Argparse int args choke on "15.0" — normalise whole floats
            if isinstance(val, float) and val == int(val):
                val = int(val)
            cmd += [flag, str(val)]

    # Boolean store_true flag — append only when truthy
    bas = params.get('brush_all_segments')
    if bas is not None and str(bas).strip() != '':
        if _validate_param('brush_all_segments', bas):
            cmd.append('--brush_all_segments')

    # Inverted booleans: the CLI defaults the new behaviours ON, so the
    # flags are negative (--no_stratify, --no_scale_features,
    # --no_hint_aware_brush, --brush_keep_intermediates). YAML keys keep
    # the positive sense so analysts read "stratify: true" not "no_stratify".
    _inverted = {
        'stratify':                ('--no_stratify',           True),
        'scale_features':          ('--no_scale_features',     True),
        'hint_aware_brush':        ('--no_hint_aware_brush',   True),
        'brush_keep_intermediates': ('--brush_keep_intermediates', False),
    }
    for key, (flag, default_on) in _inverted.items():
        v = params.get(key)
        if v is None or str(v).strip() == '':
            continue
        v = _validate_param(key, v)
        # Append the negative flag only when the user explicitly toggled
        # off a default-on behaviour, or explicitly asked for the
        # default-off one (debug intermediates).
        if default_on and not v:
            cmd.append(flag)
        if (not default_on) and v:
            cmd.append(flag)

    eb = params.get('embed_bands')
    if eb and str(eb).strip():
        eb = _validate_embed_bands(eb)
        if eb:
            # embed_bands holds ABSOLUTE 1-based indices into the stack
            # the CLI receives. When B8 is excluded that stack is
            # narrower, so the saved indices no longer name the same
            # bands -- and any index past the new count makes the
            # embedding step fail outright. Remap to the surviving
            # positions rather than passing stale numbers through.
            if _kept_band_map:
                from .band_select import remap_indices
                eb = remap_indices(
                    eb, _kept_band_map,
                    log=lambda m: fire.console_log.append(m))
            if eb:
                cmd += ['--embed_bands', eb]

    return cmd
