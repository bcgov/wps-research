"""On-disk persistence — fire_state.yaml, sessions, settings, notes, IPs.

Stateful helpers read shared state through the module-level ``state``
attribute, set by :func:`init` at server boot. Cross-module dependencies
(rebrush procs, notification push, ML metrics) are wired by ``init``.
"""

import os
import shutil
import sys
import threading
import time

from .state import AppState, FireStatus
from .io_utils import _atomic_yaml_dump

state: AppState = None

# Wired by init():
_rebrush_procs: dict = None
_rebrush_procs_lock: threading.Lock = None
_compute_agreement_cb = None
_compute_ml_area_cb = None
_push_notification_cb = None


def init(app_state: AppState,
         rebrush_procs: dict, rebrush_procs_lock: threading.Lock,
         compute_agreement_cb, compute_ml_area_cb,
         push_notification_cb):
    global state, _rebrush_procs, _rebrush_procs_lock
    global _compute_agreement_cb, _compute_ml_area_cb, _push_notification_cb
    state = app_state
    _rebrush_procs = rebrush_procs
    _rebrush_procs_lock = rebrush_procs_lock
    _compute_agreement_cb = compute_agreement_cb
    _compute_ml_area_cb = compute_ml_area_cb
    _push_notification_cb = push_notification_cb


def _save_sessions():
    """Persist sessions to disk (tokens are already hashed in memory)."""
    if not state.session_file:
        return
    try:
        with state.lock:
            snap = dict(state.sessions)
        _atomic_yaml_dump(state.session_file, snap)
    except Exception as exc:
        sys.stderr.write(f'[save] WARNING: Failed to save sessions: {exc}\n')
        sys.stderr.flush()


def _save_settings():
    """Persist recommended settings to shared_root (not per-year)."""
    try:
        settings_path = state.settings_file or os.path.join(
            state.shared_root or state.output_root,
            'recommended_settings.yaml')
        payload = {
            'k_runs_per_setting': int(state.k_runs_per_setting),
            'k_jitter': int(state.k_jitter),
            'settings': [
                {'label': str(s.get('label', '')),
                 'params': dict(s.get('params', {}))}
                for s in state.recommended_settings
            ],
        }
        _atomic_yaml_dump(settings_path, payload, mode=0o644)
    except Exception as exc:
        sys.stderr.write(
            f'[save] WARNING: Failed to save settings: {exc}\n')
        sys.stderr.flush()


def _save_notes():
    """Persist all fire notes to notes.yaml."""
    try:
        notes_path = os.path.join(state.output_root, 'notes.yaml')
        with state.lock:
            notes_data = {fn: fire.notes
                          for fn, fire in state.fires.items()
                          if fire.notes}
        _atomic_yaml_dump(notes_path, notes_data, mode=0o644)
    except Exception as exc:
        sys.stderr.write(
            f'[save] WARNING: Failed to save notes: {exc}\n')
        sys.stderr.flush()


def _save_ip_list():
    """Persist approved, blocked, and pending IPs to disk."""
    if not state.ip_file:
        return
    try:
        with state.lock:
            data = {
                'approved': dict(state.approved_ips),
                'blocked': dict(state.blocked_ips),
                'pending': dict(state.pending_ips),
            }
        _atomic_yaml_dump(state.ip_file, data)
    except Exception as exc:
        sys.stderr.write(
            f'[save] WARNING: Failed to save IP list: {exc}\n')


def _save_fire_state():
    """Persist per-fire state to fire_state.yaml so mapped fires survive restart."""
    # Refuse to overwrite a file we couldn't parse on boot. Without this
    # guard, a corrupt-but-recoverable fire_state.yaml would be clobbered
    # by the next save with the stripped init-from-GDF state, permanently
    # destroying ACCEPTED/MAPPED markers that were still in the old file.
    if state.fire_state_load_failed:
        return
    try:
        data = {}
        # Hold the lock across the entire snapshot so no fire attribute is
        # read while another thread is mutating it. state.lock is an RLock
        # so any inner helpers that re-acquire it remain safe.
        with state.lock:
            for fn, fire in state.fires.items():
                # Only persist fires with meaningful state beyond PENDING
                if (fire.status == FireStatus.PENDING and not fire.hidden
                        and not fire.cache_dir):
                    continue
                entry = {
                    'status': fire.status.value,
                    'hidden': fire.hidden,
                }
                if fire.cache_dir:
                    entry['cache_dir'] = fire.cache_dir
                if fire.crop_bin:
                    entry['crop_bin'] = fire.crop_bin
                if fire.hint_bin:
                    entry['hint_bin'] = fire.hint_bin
                if fire.perim_bin:
                    entry['perim_bin'] = fire.perim_bin
                if fire.viirs_bin:
                    entry['viirs_bin'] = fire.viirs_bin
                if fire.crop_w:
                    entry['crop_w'] = fire.crop_w
                    entry['crop_h'] = fire.crop_h
                if fire.padding_used:
                    entry['padding_used'] = fire.padding_used
                if fire.acc_start:
                    entry['acc_start'] = fire.acc_start
                    entry['acc_end'] = fire.acc_end
                if fire.perimeter_type:
                    entry['perimeter_type'] = fire.perimeter_type
                if fire.sample_size:
                    entry['sample_size'] = fire.sample_size
                if fire.available_views:
                    entry['available_views'] = list(fire.available_views)
                if fire.last_comparison:
                    entry['last_comparison'] = fire.last_comparison
                if fire.last_params:
                    entry['last_params'] = dict(fire.last_params)
                if fire.ml_area_ha >= 0:
                    entry['ml_area_ha'] = fire.ml_area_ha
                if fire.agreement_pct >= 0:
                    entry['agreement_pct'] = fire.agreement_pct
                if fire.previously_accepted:
                    entry['previously_accepted'] = True
                if fire.previously_accepted_agreement_pct >= 0:
                    entry['previously_accepted_agreement_pct'] = (
                        fire.previously_accepted_agreement_pct)
                if fire.recommended_override:
                    entry['recommended_override'] = [
                        {'label': str(s.get('label', '')),
                         'params': dict(s.get('params', {}))}
                        for s in fire.recommended_override
                    ]
                if fire.last_preset:
                    entry['last_preset'] = str(fire.last_preset)
                if fire.last_cancel_reason:
                    entry['last_cancel_reason'] = str(
                        fire.last_cancel_reason)
                # Persist serial gallery state so the results gallery
                # survives restart. Without this, fire.serial_results
                # defaults back to [] on boot and the left-pane ML overlay
                # (restored from the canonical classified.bin) appears
                # without the per-run cards.
                if fire.serial_results:
                    entry['serial_results'] = [
                        {k: v for k, v in r.items()}
                        for r in fire.serial_results
                    ]
                if fire.serial_settings:
                    entry['serial_settings'] = [
                        {'label': str(s.get('label', '')),
                         'params': dict(s.get('params', {}))}
                        for s in fire.serial_settings
                    ]
                data[fn] = entry

        state_path = os.path.join(state.output_root, 'fire_state.yaml')
        _atomic_yaml_dump(state_path, data, mode=0o644)
    except Exception as exc:
        sys.stderr.write(
            f'[save] WARNING: Failed to save fire state: {exc}\n')
        sys.stderr.flush()


def _load_fire_state():
    """Restore per-fire state from fire_state.yaml after init_fires_from_gdf."""
    state_path = os.path.join(state.output_root, 'fire_state.yaml')
    if not os.path.isfile(state_path):
        return

    try:
        import yaml
        with open(state_path) as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        # Rotate the unparseable file aside so the refuse-to-save guard
        # in _save_fire_state does not need the original path free, and
        # an operator has something to inspect/recover. Then flip the
        # load-failed flag to block future saves from clobbering the
        # on-disk original.
        backup = f'{state_path}.corrupt-{int(time.time())}'
        try:
            shutil.copy2(state_path, backup)
        except OSError:
            backup = '<copy failed>'
        sys.stderr.write(
            f'[load] CRITICAL: fire_state.yaml failed to parse ({exc}). '
            f'Copied aside to {backup}. Saves are now blocked; '
            f'investigate and restart before next save.\n')
        sys.stderr.flush()
        state.fire_state_load_failed = True
        return

    restored = 0
    for fn, entry in data.items():
        if fn not in state.fires:
            continue
        fire = state.fires[fn]

        # Restore hidden flag
        if entry.get('hidden'):
            fire.hidden = True

        # Restore per-fire recommended override (independent of cache_dir)
        override = entry.get('recommended_override')
        if isinstance(override, list) and override:
            clean = []
            for row in override:
                if not isinstance(row, dict):
                    continue
                clean.append({
                    'label': str(row.get('label', '') or ''),
                    'params': dict(row.get('params', {}) or {}),
                })
            if clean:
                fire.recommended_override = clean
        if entry.get('last_preset'):
            fire.last_preset = str(entry['last_preset'])
        if entry.get('last_cancel_reason'):
            fire.last_cancel_reason = str(entry['last_cancel_reason'])

        saved_status = entry.get('status', 'pending')

        # Don't downgrade: if init_fires_from_gdf already found ACCEPTED,
        # only restore supplementary fields, not status
        if fire.status == FireStatus.ACCEPTED and saved_status != 'accepted':
            # Just restore hidden and skip — the accepted state from disk
            # is authoritative
            continue

        # Restore cache paths — but only if they still exist on disk
        cache_dir = entry.get('cache_dir', '')
        if cache_dir and os.path.isdir(cache_dir):
            fire.cache_dir = cache_dir
            fire.crop_bin = entry.get('crop_bin', '')
            fire.hint_bin = entry.get('hint_bin', '')
            fire.perim_bin = entry.get('perim_bin', '')
            fire.viirs_bin = entry.get('viirs_bin', '')
            fire.crop_w = entry.get('crop_w', 0)
            fire.crop_h = entry.get('crop_h', 0)
            fire.padding_used = entry.get('padding_used', 0.0)
            fire.acc_start = entry.get('acc_start', '')
            fire.acc_end = entry.get('acc_end', '')
            fire.perimeter_type = entry.get('perimeter_type', '')
            fire.sample_size = entry.get('sample_size', 0)
            # Drop view keys whose preview PNG is gone from disk.
            # Manual .web_cache wipes leave fire_state.yaml claiming
            # views that no longer exist; the UI then asks for them and
            # gets a 404 ("View ... not available"). Filter here so all
            # callers see a self-consistent list.
            saved_views = entry.get('available_views', []) or []
            previews_dir = os.path.join(cache_dir, 'previews')
            fire.available_views = [
                v for v in saved_views
                if os.path.isfile(os.path.join(previews_dir, f'{v}.png'))
            ]
            fire.last_comparison = entry.get('last_comparison', '')
            fire.last_params = entry.get('last_params', {})
            fire.ml_area_ha = entry.get('ml_area_ha', -1.0)
            fire.agreement_pct = entry.get('agreement_pct', -1.0)
            fire.previously_accepted = entry.get(
                'previously_accepted', False)
            fire.previously_accepted_agreement_pct = float(
                entry.get('previously_accepted_agreement_pct', -1.0))

            # Restore serial gallery. Each entry's classified.bin must
            # still be on disk — drop entries whose files were deleted
            # out of band (manual cleanup, external wipe). The 'previous
            # accepted snapshot' run (is_previous=True) is restored
            # unconditionally; its backing files live in accepted output,
            # not the .web_cache directory.
            saved_serial = entry.get('serial_results')
            if isinstance(saved_serial, list) and saved_serial:
                clean_serial = []
                for r in saved_serial:
                    if not isinstance(r, dict):
                        continue
                    clf = r.get('classified', '')
                    if r.get('is_previous'):
                        clean_serial.append(dict(r))
                        continue
                    if clf and os.path.isfile(clf):
                        clean_serial.append(dict(r))
                if clean_serial:
                    fire.serial_results = clean_serial

            saved_settings = entry.get('serial_settings')
            if isinstance(saved_settings, list) and saved_settings:
                clean_settings = []
                for s in saved_settings:
                    if not isinstance(s, dict):
                        continue
                    clean_settings.append({
                        'label': str(s.get('label', '') or ''),
                        'params': dict(s.get('params', {}) or {}),
                    })
                if clean_settings:
                    fire.serial_settings = clean_settings

            # Fallback: if fire_state.yaml predates the serial_results
            # persistence change, reconstruct a partial gallery by
            # scanning the cache dir for per-run classified.bin files.
            # Setting labels and params are lost (they lived only in
            # memory), but run_id, file paths, agreement_pct, and
            # ml_area_ha can be recovered. Accept/Rebrush still work.
            if not fire.serial_results:
                try:
                    import re as _re
                    pat = _re.compile(
                        r'^' + _re.escape(fn)
                        + r'_serial_(\d+)_classified\.bin$')
                    scanned = []
                    for name in os.listdir(cache_dir):
                        m = pat.match(name)
                        if not m:
                            continue
                        rid = int(m.group(1))
                        clf = os.path.join(cache_dir, name)
                        comp = os.path.join(
                            cache_dir, f'{fn}_serial_{rid}.png')
                        entry_ = {
                            'run_id': rid,
                            'setting_idx': 0,
                            'run_idx': 0,
                            'setting_label': '',
                            'params': {},
                            'classified': clf,
                            'comparison': comp if os.path.isfile(comp)
                                          else '',
                            'agreement_pct': _compute_agreement_cb(
                                fire, clf_path=clf),
                            'ml_area_ha': _compute_ml_area_cb(fire, clf),
                        }
                        scanned.append(entry_)
                    if scanned:
                        scanned.sort(key=lambda r: r['run_id'])
                        fire.serial_results = scanned
                        sys.stderr.write(
                            f'[load] Rebuilt {len(scanned)} serial '
                            f'gallery entries for {fn} from cache scan '
                            f'(no persisted serial_results).\n')
                except Exception as exc:
                    sys.stderr.write(
                        f'[load] WARNING: serial gallery rebuild '
                        f'failed for {fn}: {exc}\n')
                    sys.stderr.flush()

            # Validate critical files exist before restoring status
            if saved_status in ('ready', 'mapped'):
                crop_ok = (fire.crop_bin
                           and os.path.isfile(fire.crop_bin))
                hint_ok = (fire.hint_bin
                           and os.path.isfile(fire.hint_bin))
                if crop_ok and hint_ok:
                    fire.status = FireStatus(saved_status)
                    restored += 1
                else:
                    fire.status = FireStatus.PENDING
            elif saved_status == 'accepted':
                fire.status = FireStatus.ACCEPTED
                restored += 1
            # MAPPING/PREPARING on disk means crashed mid-work — reset
            elif saved_status in ('mapping', 'preparing'):
                # Check if a mapped result exists in cache
                comp = os.path.join(
                    cache_dir, f'{fn}_comparison.png')
                if os.path.isfile(comp):
                    fire.status = FireStatus.MAPPED
                    fire.last_comparison = comp
                    restored += 1
                else:
                    fire.status = FireStatus.READY
        elif saved_status == 'accepted':
            fire.status = FireStatus.ACCEPTED

    if restored:
        sys.stderr.write(
            f'[load] Restored state for {restored} fire(s) '
            f'from fire_state.yaml\n')
        sys.stderr.flush()


def _save_active_year():
    """Persist the currently-active year so it survives restarts."""
    if not state.shared_root:
        return
    try:
        path = os.path.join(state.shared_root, 'active_year.yaml')
        _atomic_yaml_dump(path, {'active_year': int(state.active_year)},
                          mode=0o644)
    except Exception as exc:
        sys.stderr.write(
            f'[save] WARNING: Failed to save active year: {exc}\n')
        sys.stderr.flush()


def _switch_year(year: int) -> tuple[bool, str]:
    """Re-point active-year state to *year* without restarting the server.

    Fails fast if any long-running job is in flight (one-shot mapping,
    batch mapping, rebrush). Returns (ok, message). On success the caller
    should persist and notify clients to reload.
    """
    if not isinstance(year, int):
        return False, 'year must be an int'
    if year not in state.rasters_by_year:
        return False, f'unknown year {year}'
    if year == state.active_year:
        return True, 'already active'

    # Busy checks — refuse mid-job rather than corrupting in-flight state.
    with state.lock:
        if state.current_job is not None:
            return False, ('A mapping job is running '
                           f'(fire {state.current_job.get("fire_numbe")}). '
                           'Wait for it to finish or cancel it.')
        if state.batch_status and state.batch_status.get('running'):
            return False, ('A batch mapping is running '
                           f'(fire {state.batch_status.get("current_fire") or "?"}). '
                           'Cancel it from the fire list or wait.')
    # Rebrush workers run outside _gpu_lock — check explicitly so the
    # switch doesn't wipe fire_state.yaml out from under a running
    # class_brush.exe.
    with _rebrush_procs_lock:
        if _rebrush_procs:
            active = ', '.join(sorted(_rebrush_procs.keys())[:3])
            more = ('' if len(_rebrush_procs) <= 3
                    else f' (+{len(_rebrush_procs) - 3} more)')
            return False, (f'A rebrush is running on fire {active}{more}. '
                           'Cancel it or wait.')

    from batch_fire_mapping.run_fire_mapping import (
        get_raster_info, load_all_viirs)
    import geopandas as _gpd
    from shapely.geometry import box as _shapely_box

    new_raster = state.rasters_by_year[year]
    new_outdir = state.outdirs_by_year[year]
    new_viirs  = state.viirs_shp_dirs_by_year[year]

    # Persist current year's state before swapping so nothing is lost.
    _save_fire_state()

    try:
        crs_wkt, gt, W, H = get_raster_info(new_raster)
    except Exception as exc:
        return False, f'Failed to read raster: {exc}'

    # Re-project + spatial-filter the cached raw polygon GDF to the new
    # raster. Falling back to re-reading the shapefile if the cache is
    # missing (e.g. a stale AppState).
    raw = state.polygon_gdf_raw
    if raw is None:
        try:
            raw = _gpd.read_file(state.polygon_file)
            state.polygon_gdf_raw = raw
        except Exception as exc:
            return False, f'Failed to read polygons: {exc}'

    from batch_fire_mapping.run_fire_mapping import (
        _wkt_to_pyproj_crs, raster_native_extent)
    import pandas as _pd
    try:
        # Filter by FIRE_YEAR first (cheap, drops most rows before reproject)
        if 'FIRE_YEAR' in raw.columns:
            fy = _pd.to_numeric(raw['FIRE_YEAR'], errors='coerce')
            gdf = raw[fy == year].copy()
        else:
            gdf = raw.copy()
        if gdf.empty:
            return False, (f'No polygons with FIRE_YEAR={year} in the '
                           'shapefile.')
        gdf = gdf.to_crs(_wkt_to_pyproj_crs(crs_wkt))
        xmin, ymin, xmax, ymax = raster_native_extent(gt, W, H)
        gdf = gdf[gdf.geometry.intersects(
            _shapely_box(xmin, ymin, xmax, ymax))].copy()
    except Exception as exc:
        return False, f'Failed to filter polygons: {exc}'

    if gdf.empty:
        return False, (f'No polygons with FIRE_YEAR={year} overlap the '
                       f'year-{year} raster extent.')

    # Load VIIRS for the new year (traditional mode -> empty GDF)
    if state.perimeter_mode == 'traditional':
        viirs_gdf = _gpd.GeoDataFrame()
    else:
        try:
            viirs_gdf = load_all_viirs(new_viirs, crs_wkt)
        except Exception as exc:
            return False, f'Failed to load VIIRS: {exc}'

    # Atomic swap under the state lock. Rebuild fires from the new gdf;
    # _load_fire_state reads the new outdir's fire_state.yaml afterwards.
    with state.lock:
        state.active_year    = year
        state.raster_path    = new_raster
        state.output_root    = new_outdir
        state.viirs_shp_dir  = new_viirs
        state.raster_crs     = crs_wkt
        state.raster_gt      = gt
        state.raster_W       = W
        state.raster_H       = H
        state.gdf            = gdf
        state.viirs_gdf      = viirs_gdf
        state.fires          = {}
        state.init_fires_from_gdf()

    _load_fire_state()
    _save_active_year()

    sys.stderr.write(
        f'[switch_year] Active year -> {year}, {len(state.fires)} fire(s)\n')
    sys.stderr.flush()
    _push_notification_cb(
        None, 'info',
        f'Active year switched to {year}',
        f'{len(state.fires)} fire(s) loaded. Reload the page to see the '
        'new fire list.',
        action={'url': '/', 'label': 'Reload fire list'})
    return True, 'ok'
