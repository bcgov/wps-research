"""
batch_fire_mapping_web
======================
Web interface for interactive fire mapping, multi-year aware.

Launch
------
    python -m batch_fire_mapping_web  POLYGONS.shp  \\
        --rasters  pgfc_2022.bin  pgfc_2023.bin  pgfc_2024.bin  \\
        --out_root /path/to/mother_dir  [options]

Then open http://localhost:8765 in a browser. Admins can switch the
active year from the UI; each year gets its own per-year output dir
``<out_root>/<raster_stem>_mapping_results`` automatically.
"""

# ---------------------------------------------------------------------------
# Path setup — identical to batch_fire_mapping/run_fire_mapping.py
# ---------------------------------------------------------------------------
import os
import re
import sys

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)                           # data/bill/
_REPO_ROOT    = os.path.dirname(os.path.dirname(_PROJECT_ROOT))  # wps-research/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import datetime

import geopandas as gpd
import pandas as pd

# Project imports (via sys.path)
from batch_fire_mapping.run_fire_mapping import (
    get_raster_info,
    load_and_filter_polygons,
    load_all_viirs,
    download_viirs,
    shapify_viirs,
    load_token,
)


def _year_from_filename(path: str) -> int:
    """Extract a 4-digit year from a raster filename.

    Scans all 4-digit substrings in the file's stem and keeps those in
    [1970, now+1]. Exactly one plausible year must be derivable; raises
    ValueError otherwise. This makes ``pgfc_2023.bin`` -> 2023 and
    ``S2C_MSIL1C_20251014T192401_20m.bin`` -> 2025 without hardcoding
    digit positions.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    now_year = datetime.datetime.now().year
    lo, hi = 1970, now_year + 1
    found = set()
    for m in re.finditer(r'(?=(\d{4}))', stem):
        try:
            y = int(m.group(1))
        except ValueError:
            continue
        if lo <= y <= hi:
            found.add(y)
    if len(found) == 0:
        raise ValueError(
            f'Cannot find a 4-digit year in [{lo},{hi}] in '
            f'filename "{stem}".')
    if len(found) > 1:
        raise ValueError(
            f'Filename "{stem}" contains multiple year-like tokens '
            f'{sorted(found)} — cannot pick one automatically.')
    return next(iter(found))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='batch_fire_mapping_web',
        description='Web interface for interactive Sentinel-2 fire mapping '
                    '(multi-year).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example
-------
  python -m batch_fire_mapping_web  \\
      IN_HISTORICAL_FIRE_POLYGONS_SVW.shp  \\
      --rasters  pgfc_2022.bin  pgfc_2023.bin  pgfc_2024.bin  \\
      --out_root /data/bill/mapping_results \\
      --skip_download --padding 0.2
        """,
    )

    # Required
    p.add_argument('polygon_file',
                   help='Fire perimeters shapefile (.shp)')
    p.add_argument('--rasters', nargs='+', required=True,
                   help='One or more Sentinel-2 ENVI .bin rasters; each '
                        'filename must contain a unique 4-digit year.')
    p.add_argument('--out_root', required=True,
                   help='Mother directory; per-year outputs go to '
                        '<out_root>/<raster_stem>_mapping_results.')
    p.add_argument('--year', type=int, default=None,
                   help='Initial active year (default: value stored in '
                        '<out_root>/active_year.yaml, else newest year).')

    # Perimeter mode
    p.add_argument('--perimeter_mode', default='viirs',
                   choices=['viirs', 'traditional'],
                   help='Hint source: viirs (default) or traditional')

    # VIIRS
    p.add_argument('--skip_download', action='store_true',
                   help='Skip VIIRS download and shapify')
    p.add_argument('--shapify_workers', type=int, default=8,
                   help='Parallel shapify workers (default: 8)')

    # Sampling defaults
    p.add_argument('--sample_rate', type=float, default=0.05,
                   help='Default sample rate (default: 0.05)')
    p.add_argument('--min_samples', type=int, default=500)
    p.add_argument('--max_samples', type=int, default=30000)

    # Server
    p.add_argument('--host', default='0.0.0.0',
                   help='Server bind address (default: 0.0.0.0). '
                        'Use 127.0.0.1 to restrict to localhost only.')
    p.add_argument('--port', type=int, default=8765,
                   help='Server port (default: 8765)')

    # Authentication
    p.add_argument('--admin_password', default=None,
                   help='Admin password (or env FIRE_ADMIN_PASSWORD)')
    p.add_argument('--user_password', default=None,
                   help='Generic user password (or env FIRE_USER_PASSWORD)')
    p.add_argument('--insecure_no_auth', action='store_true',
                   help='Allow running without passwords (opt-in)')
    p.add_argument('--trust_proxy', action='store_true',
                   help='Trust X-Forwarded-For header for client IP '
                        '(use only behind a trusted reverse proxy)')

    return p


def _prepare_year_for_viirs(raster_path: str, polygon_file: str,
                            viirs_save_dir: str, viirs_shp_dir: str,
                            args):
    """Download + shapify VIIRS for a given year's raster, as needed.

    Mirrors the original single-raster path. Safe to call per-year during
    startup; each year has its own viirs_save_dir so nothing collides.
    """
    os.makedirs(viirs_shp_dir, exist_ok=True)

    if args.perimeter_mode == 'traditional':
        print(f'      [{os.path.basename(raster_path)}] '
              f'perimeter_mode=traditional — skipping VIIRS')
        return

    if args.skip_download:
        print(f'      [{os.path.basename(raster_path)}] '
              f'--skip_download — skipping VIIRS download/shapify')
        return

    # Load polygons for this year to figure out the date window
    gdf = load_and_filter_polygons(polygon_file, raster_path)
    if gdf.empty:
        print(f'      [{os.path.basename(raster_path)}] '
              f'no polygons in raster extent — skipping VIIRS download')
        return

    years = pd.to_numeric(
        gdf.get('FIRE_YEAR', pd.Series(dtype=int)),
        errors='coerce').dropna()
    if years.empty:
        years = pd.to_datetime(
            gdf['FIRE_DATE'], errors='coerce').dt.year.dropna()
    if years.empty:
        print(f'      [{os.path.basename(raster_path)}] '
              f'cannot determine fire years — skipping VIIRS download')
        return

    min_y, max_y = int(years.min()), int(years.max())
    dl_start = datetime.datetime(min_y, 1, 1)
    dl_end   = datetime.datetime(max_y, 12, 31)

    print(f'      [{os.path.basename(raster_path)}] VIIRS '
          f'{dl_start.date()} -> {dl_end.date()}')
    token = load_token()
    download_viirs(raster_path, dl_start, dl_end, token, viirs_save_dir)
    shapify_viirs(viirs_save_dir, raster_path, args.shapify_workers)


def main():
    args = _build_parser().parse_args()

    polygon_file = os.path.abspath(args.polygon_file)
    out_root     = os.path.abspath(args.out_root)

    if not os.path.exists(polygon_file):
        sys.exit(f'ERROR: Polygon file not found: {polygon_file}')
    os.makedirs(out_root, exist_ok=True)

    # ------------------------------------------------------------------
    # Build {year -> raster} registry with filename-based detection
    # ------------------------------------------------------------------
    rasters_abs = [os.path.abspath(r) for r in args.rasters]
    for r in rasters_abs:
        if not os.path.exists(r):
            sys.exit(f'ERROR: Raster not found: {r}')

    rasters_by_year: dict = {}
    for r in rasters_abs:
        try:
            y = _year_from_filename(r)
        except ValueError as e:
            sys.exit(f'ERROR: {e}')
        if y in rasters_by_year:
            sys.exit(
                f'ERROR: Year {y} appears in two rasters:\n'
                f'  {rasters_by_year[y]}\n  {r}\n'
                f'Each year must be unique.')
        rasters_by_year[y] = r

    # Per-year output dirs: <out_root>/<raster_stem>_mapping_results
    outdirs_by_year: dict = {}
    viirs_shp_dirs_by_year: dict = {}
    for y, r in rasters_by_year.items():
        stem = os.path.splitext(os.path.basename(r))[0]
        od = os.path.join(out_root, f'{stem}_mapping_results')
        os.makedirs(od, exist_ok=True)
        outdirs_by_year[y] = od

        raster_dir = os.path.dirname(r)
        viirs_save_dir = os.path.join(raster_dir, f'{stem}_VIIRS')
        viirs_shp_dirs_by_year[y] = os.path.join(viirs_save_dir, 'VNP14IMG')

    # ------------------------------------------------------------------
    # Decide initial active year
    # ------------------------------------------------------------------
    import yaml
    active_year_file = os.path.join(out_root, 'active_year.yaml')
    active_year = None
    if args.year is not None:
        if args.year not in rasters_by_year:
            sys.exit(f'ERROR: --year {args.year} not in '
                     f'{sorted(rasters_by_year)}')
        active_year = args.year
    else:
        if os.path.isfile(active_year_file):
            try:
                with open(active_year_file) as _f:
                    _d = yaml.safe_load(_f) or {}
                cand = int(_d.get('active_year', 0))
                if cand in rasters_by_year:
                    active_year = cand
            except Exception:
                pass
        if active_year is None:
            active_year = max(rasters_by_year)  # newest

    raster_path    = rasters_by_year[active_year]
    output_root    = outdirs_by_year[active_year]
    viirs_shp_dir  = viirs_shp_dirs_by_year[active_year]

    sep = '=' * 60
    print(f'\n{sep}')
    print('  BATCH FIRE MAPPING — WEB INTERFACE (multi-year)')
    print(sep)
    print(f'  Polygons   : {polygon_file}')
    print(f'  Out root   : {out_root}')
    print(f'  Years      : {sorted(rasters_by_year)}')
    print(f'  Active year: {active_year}')
    print(f'  Raster     : {raster_path}')
    print(f'  Output     : {output_root}')
    print(f'  Perimeter  : {args.perimeter_mode}')
    print(sep)

    # ------------------------------------------------------------------
    # Step 1 — VIIRS download/shapify for every year (so switches are fast)
    # ------------------------------------------------------------------
    print('\n[1/4] VIIRS prepare (per year) ...')
    for y in sorted(rasters_by_year):
        r = rasters_by_year[y]
        raster_dir = os.path.dirname(r)
        stem = os.path.splitext(os.path.basename(r))[0]
        viirs_save_dir = os.path.join(raster_dir, f'{stem}_VIIRS')
        _prepare_year_for_viirs(
            r, polygon_file, viirs_save_dir,
            viirs_shp_dirs_by_year[y], args)

    # ------------------------------------------------------------------
    # Step 2 — Load polygons + filter for the active year's raster.
    # Filter by FIRE_YEAR == active_year so fires from other years (that
    # happen to intersect this raster's footprint) don't leak into the
    # list. Each year's raster is meant to show only that year's fires.
    # ------------------------------------------------------------------
    print('\n[2/4] Loading polygons ...')
    gdf = load_and_filter_polygons(polygon_file, raster_path,
                                   year=active_year)
    if gdf.empty:
        sys.exit(f'No polygons with FIRE_YEAR={active_year} inside the '
                 'active raster extent.')
    print(f'      {len(gdf)} fire(s) loaded for year {active_year}.')

    # Raw GDF (source CRS, unfiltered) — cached for fast year switches
    raw_gdf = gpd.read_file(polygon_file)

    # ------------------------------------------------------------------
    # Step 3 — Load VIIRS shapefiles for the active year
    # ------------------------------------------------------------------
    viirs_gdf = gpd.GeoDataFrame()
    if args.perimeter_mode != 'traditional':
        print('\n[3/4] Loading VIIRS shapefiles ...')
        crs_wkt, _, _, _ = get_raster_info(raster_path)
        viirs_gdf = load_all_viirs(viirs_shp_dir, crs_wkt)

    # ------------------------------------------------------------------
    # Step 4 — Initialise application state and start server
    # ------------------------------------------------------------------
    print('\n[4/4] Starting web server ...')

    crs_wkt, gt, W, H = get_raster_info(raster_path)

    from .state import AppState
    from .app import init_app, create_server

    app_state = AppState()
    # Active-year views
    app_state.gdf            = gdf
    app_state.viirs_gdf      = viirs_gdf
    app_state.raster_path    = raster_path
    app_state.polygon_file   = polygon_file
    app_state.raster_crs     = crs_wkt
    app_state.raster_gt      = gt
    app_state.raster_W       = W
    app_state.raster_H       = H
    app_state.viirs_shp_dir  = viirs_shp_dir
    app_state.output_root    = output_root

    # Multi-year registry
    app_state.active_year             = active_year
    app_state.shared_root             = out_root
    app_state.rasters_by_year         = rasters_by_year
    app_state.outdirs_by_year         = outdirs_by_year
    app_state.viirs_shp_dirs_by_year  = viirs_shp_dirs_by_year
    app_state.polygon_gdf_raw         = raw_gdf

    app_state.project_root   = _PROJECT_ROOT
    app_state.cli_script     = os.path.join(
        _REPO_ROOT, 'py', 'fire_mapping', 'fire_mapping_cli.py')
    app_state.padding        = 0.1
    app_state.sample_rate    = args.sample_rate
    app_state.min_samples    = args.min_samples
    app_state.max_samples    = args.max_samples
    app_state.perimeter_mode = args.perimeter_mode
    app_state.admin_password = (args.admin_password
                                or os.environ.get('FIRE_ADMIN_PASSWORD'))
    app_state.user_password  = (args.user_password
                                or os.environ.get('FIRE_USER_PASSWORD'))

    app_state.trust_proxy = args.trust_proxy
    app_state.insecure_no_auth = args.insecure_no_auth

    # Validate password configuration
    if (not app_state.admin_password and not app_state.user_password
            and not app_state.insecure_no_auth):
        sys.exit(
            'ERROR: No passwords configured. Pass --admin_password / '
            '--user_password (or set FIRE_ADMIN_PASSWORD / '
            'FIRE_USER_PASSWORD), or pass --insecure_no_auth to '
            'run without authentication. See README.')
    if app_state.user_password and not app_state.admin_password:
        sys.exit('ERROR: --user_password requires --admin_password. '
                 'Without an admin, no one can approve user IPs.')
    if (app_state.admin_password and app_state.user_password
            and app_state.admin_password == app_state.user_password):
        sys.exit('ERROR: --admin_password and --user_password must be '
                 'different. Otherwise all users become admin.')

    # Compute allowed origins for CSRF checks
    app_state.allowed_origins = {
        f'http://localhost:{args.port}',
        f'http://127.0.0.1:{args.port}',
    }
    if args.host not in ('127.0.0.1', 'localhost'):
        app_state.allowed_origins.add(f'http://{args.host}:{args.port}')

    if not os.path.isfile(app_state.cli_script):
        sys.exit(f'ERROR: fire_mapping_cli.py not found at '
                 f'{app_state.cli_script}')

    # Load recommended settings from shared_root (shared across all years)
    # with a fallback to the package default.
    # Schema: {k_runs_per_setting, k_jitter, settings: [{label, params}, ...]}
    app_state.settings_file = os.path.join(out_root, 'recommended_settings.yaml')
    _pkg_settings  = os.path.join(_HERE, 'recommended_settings.yaml')
    _settings_path = (app_state.settings_file
                      if os.path.isfile(app_state.settings_file)
                      else (_pkg_settings if os.path.isfile(_pkg_settings)
                            else None))
    if _settings_path is None:
        sys.exit('ERROR: recommended_settings.yaml not found in out_root '
                 'or package dir.')
    try:
        with open(_settings_path) as _f:
            _cfg = yaml.safe_load(_f)
    except Exception as _e:
        sys.exit(f'ERROR: Failed to read {_settings_path}: {_e}')

    if isinstance(_cfg, list):
        sys.exit(
            f'ERROR: {_settings_path} uses the legacy size-bucket schema '
            '(list with min_ha/max_ha). The new schema is a dict with '
            'keys k_runs_per_setting, k_jitter, and settings. Regenerate '
            'the file; see package default for a template.')
    if not isinstance(_cfg, dict) or 'settings' not in _cfg:
        sys.exit(
            f'ERROR: {_settings_path} missing required key "settings". '
            'Expected: {k_runs_per_setting, k_jitter, settings: [...]}.')

    _settings_list = _cfg.get('settings') or []
    if not isinstance(_settings_list, list) or len(_settings_list) == 0:
        sys.exit(f'ERROR: {_settings_path} has empty "settings" list.')
    for _i, _s in enumerate(_settings_list):
        if not isinstance(_s, dict) or 'params' not in _s:
            sys.exit(f'ERROR: {_settings_path} settings[{_i}] missing '
                     '"params".')
        if 'label' not in _s or not str(_s['label']).strip():
            _s['label'] = f'setting_{_i}'

    app_state.recommended_settings = _settings_list
    try:
        app_state.k_runs_per_setting = int(_cfg.get('k_runs_per_setting', 3))
    except (TypeError, ValueError):
        app_state.k_runs_per_setting = 3
    app_state.k_runs_per_setting = max(1, min(10, app_state.k_runs_per_setting))
    try:
        app_state.k_jitter = int(_cfg.get('k_jitter', 1))
    except (TypeError, ValueError):
        app_state.k_jitter = 1
    app_state.k_jitter = max(0, app_state.k_jitter)

    print(f'      Loaded {len(app_state.recommended_settings)} '
          f'recommended setting(s) from '
          f'{"out_root" if _settings_path == app_state.settings_file else "package"}'
          f'. K={app_state.k_runs_per_setting}, jitter={app_state.k_jitter}')

    # Shared persistent files live under out_root so logins and IP rules
    # survive a year-switch (per-year fire data stays in the per-year outdir).
    app_state.ip_file = os.path.join(out_root, 'access_control.yaml')
    if os.path.isfile(app_state.ip_file):
        try:
            with open(app_state.ip_file) as _f:
                _ip_data = yaml.safe_load(_f) or {}
            app_state.approved_ips = _ip_data.get('approved', {})
            app_state.blocked_ips = _ip_data.get('blocked', {})
            app_state.pending_ips = _ip_data.get('pending', {})
            print(f'      Loaded {len(app_state.approved_ips)} approved, '
                  f'{len(app_state.blocked_ips)} blocked, '
                  f'{len(app_state.pending_ips)} pending IP(s).')
        except Exception as _e:
            print(f'      WARNING: Failed to load IP list: {_e}')

    app_state.session_file = os.path.join(out_root, 'sessions.yaml')
    if os.path.isfile(app_state.session_file):
        try:
            with open(app_state.session_file) as _f:
                _sess = yaml.safe_load(_f) or {}
            _now = datetime.datetime.now()
            for _tok, _info in list(_sess.items()):
                try:
                    _created = datetime.datetime.fromisoformat(
                        _info['created_at'])
                    if (_now - _created).total_seconds() > 30 * 86400:
                        del _sess[_tok]
                except (KeyError, ValueError):
                    del _sess[_tok]
            app_state.sessions = _sess
            print(f'      Loaded {len(app_state.sessions)} active session(s).')
        except Exception as _e:
            print(f'      WARNING: Failed to load sessions: {_e}')

    app_state.init_fires_from_gdf()

    init_app(app_state)

    # Restore per-fire state from previous session for the active year
    from .app import _load_fire_state, _save_active_year
    _load_fire_state()
    _save_active_year()

    # Parameter analyzer (admin-only). Registers its own routes on
    # FireHandler; leaves app.py untouched.
    from .analyzer_app import init_analyzer, register_routes
    register_routes()
    init_analyzer(app_state)

    server = create_server(args.host, args.port)

    print(f'\n{sep}')
    print(f'  Server ready!')
    print(f'  Local:   http://localhost:{args.port}')
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        lan_ip = s.getsockname()[0]
        s.close()
        hostname = socket.gethostname()
        print(f'  Network: http://{lan_ip}:{args.port}')
        print(f'           http://{hostname}:{args.port}')
        # Add discovered network addresses to allowed origins
        app_state.allowed_origins.add(f'http://{lan_ip}:{args.port}')
        app_state.allowed_origins.add(f'http://{hostname}:{args.port}')
    except Exception:
        pass
    if app_state.admin_password:
        print(f'  Auth:    admin + user passwords configured')
        print(f'  IP ctrl: {app_state.ip_file}')
    else:
        print(f'  Auth:    NONE (--insecure_no_auth)')
        print(f'  WARNING: All users have full admin access!')
    print(f'  Years:   {sorted(app_state.rasters_by_year)} '
          f'(active={app_state.active_year})')
    print(f'  {len(app_state.fires)} fire(s) available')
    print(f'{sep}\n')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down...')
        server.shutdown()


if __name__ == '__main__':
    main()
