"""
batch_fire_mapping_web
======================
Web interface for interactive fire mapping.

Launch
------
    python -m batch_fire_mapping_web  POLYGONS.shp  RASTER.bin  [options]

Then open http://localhost:8765 in a browser.
"""

# ---------------------------------------------------------------------------
# Path setup — identical to batch_fire_mapping/run_fire_mapping.py
# ---------------------------------------------------------------------------
import os
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='batch_fire_mapping_web',
        description='Web interface for interactive Sentinel-2 fire mapping.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example
-------
  python -m batch_fire_mapping_web  \\
      IN_HISTORICAL_FIRE_POLYGONS_SVW.shp  \\
      S2C_MSIL1C_20251014T192401_20m.bin   \\
      --skip_download --padding 0.2
        """,
    )

    # Required
    p.add_argument('polygon_file',
                   help='Fire perimeters shapefile (.shp)')
    p.add_argument('raster_file',
                   help='Sentinel-2 ENVI .bin raster')

    # Output
    p.add_argument('--out_dir', default=None,
                   help='Output root directory (default: same as raster)')

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


def main():
    args = _build_parser().parse_args()

    raster_path  = os.path.abspath(args.raster_file)
    polygon_file = os.path.abspath(args.polygon_file)
    raster_dir   = os.path.dirname(raster_path)
    output_root  = os.path.abspath(args.out_dir) if args.out_dir else raster_dir

    if not os.path.exists(raster_path):
        sys.exit(f'ERROR: Raster not found: {raster_path}')
    if not os.path.exists(polygon_file):
        sys.exit(f'ERROR: Polygon file not found: {polygon_file}')

    os.makedirs(output_root, exist_ok=True)

    sep = '=' * 60
    print(f'\n{sep}')
    print('  BATCH FIRE MAPPING — WEB INTERFACE')
    print(sep)
    print(f'  Raster     : {raster_path}')
    print(f'  Polygons   : {polygon_file}')
    print(f'  Output     : {output_root}')
    print(f'  Perimeter  : {args.perimeter_mode}')
    print(sep)

    # ------------------------------------------------------------------
    # Step 1 — Load and filter polygons
    # ------------------------------------------------------------------
    print('\n[1/4] Loading polygons ...')
    gdf = load_and_filter_polygons(polygon_file, raster_path)
    if gdf.empty:
        sys.exit('No matching polygons found in shapefile.')
    print(f'      {len(gdf)} fire(s) loaded.')

    # ------------------------------------------------------------------
    # Step 2 — VIIRS download & shapify
    # ------------------------------------------------------------------
    raster_basename = os.path.splitext(os.path.basename(raster_path))[0]
    viirs_save_dir  = os.path.join(raster_dir, f'{raster_basename}_VIIRS')
    viirs_shp_dir   = os.path.join(viirs_save_dir, 'VNP14IMG')
    os.makedirs(viirs_shp_dir, exist_ok=True)

    viirs_gdf = gpd.GeoDataFrame()

    if args.perimeter_mode == 'traditional':
        print('\n[2/4] Skipping VIIRS (--perimeter_mode=traditional)')
    elif not args.skip_download:
        years = pd.to_numeric(
            gdf.get('FIRE_YEAR', pd.Series(dtype=int)),
            errors='coerce').dropna()
        if years.empty:
            years = pd.to_datetime(
                gdf['FIRE_DATE'], errors='coerce').dt.year.dropna()
        if years.empty:
            sys.exit('ERROR: Cannot determine fire years.')

        min_y, max_y = int(years.min()), int(years.max())
        dl_start = datetime.datetime(min_y, 1, 1)
        dl_end   = datetime.datetime(max_y, 12, 31)

        print(f'\n[2/4] Downloading VIIRS '
              f'({dl_start.date()} -> {dl_end.date()}) ...')
        token = load_token()
        download_viirs(raster_path, dl_start, dl_end,
                       token, viirs_save_dir)

        print('\n[3/4] Shapifying ...')
        shapify_viirs(viirs_save_dir, raster_path,
                      args.shapify_workers)
    else:
        print('\n[2/4] Skipping VIIRS download (--skip_download)')

    # ------------------------------------------------------------------
    # Step 3 — Load all VIIRS shapefiles
    # ------------------------------------------------------------------
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

    # Load recommended settings (output_root first, fallback to package dir).
    # Schema: {k_runs_per_setting, k_jitter, settings: [{label, params}, ...]}
    # The old schema (list with min_ha/max_ha) is rejected explicitly so
    # stale configs cause a clean failure instead of silent misbehavior.
    import yaml
    _settings_path = None
    _user_settings = os.path.join(output_root, 'recommended_settings.yaml')
    _pkg_settings  = os.path.join(_HERE, 'recommended_settings.yaml')
    if os.path.isfile(_user_settings):
        _settings_path = _user_settings
    elif os.path.isfile(_pkg_settings):
        _settings_path = _pkg_settings
    if _settings_path is None:
        sys.exit('ERROR: recommended_settings.yaml not found in output dir '
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
          f'{"output dir" if _settings_path == _user_settings else "package"}.'
          f' K={app_state.k_runs_per_setting}, jitter={app_state.k_jitter}')

    # Load persistent IP access list
    app_state.ip_file = os.path.join(output_root, 'access_control.yaml')
    if os.path.isfile(app_state.ip_file):
        try:
            import yaml
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

    # Load persistent sessions
    app_state.session_file = os.path.join(output_root, 'sessions.yaml')
    if os.path.isfile(app_state.session_file):
        try:
            import yaml
            with open(app_state.session_file) as _f:
                _sess = yaml.safe_load(_f) or {}
            # Filter out expired sessions (30 days)
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

    # Restore per-fire state from previous session
    from .app import _load_fire_state
    _load_fire_state()

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
    print(f'  {len(app_state.fires)} fire(s) available')
    print(f'{sep}\n')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down...')
        server.shutdown()


if __name__ == '__main__':
    main()
