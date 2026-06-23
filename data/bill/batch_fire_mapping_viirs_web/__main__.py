"""
batch_fire_mapping_viirs_web
============================
User-defined wildfire-mapping web app, multi-year aware.

Analysts open ``/new_fire``, draw a bounding box on the year's
overview, set a fire name + start/end dates, and the server downloads
VIIRS data for that bbox+range, accumulates / rasterizes it, derives a
tight crop from the rasterized fire pixels, then seeds the standard ML
mapping pipeline.

Launch
------
    python -m batch_fire_mapping_viirs_web                          \\
        --rasters  pgfc_2022.bin  pgfc_2023.bin  pgfc_2024.bin     \\
        --out_root /path/to/mother_dir  [options]

Then open http://localhost:8765 in a browser.

Requires a LAADS DAAC token at ``/data/.tokens/laads`` (single user,
shared across the server)."""

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

# Project imports (via sys.path)
from batch_fire_mapping.run_fire_mapping import get_raster_info


_LAADS_TOKEN_PATH = '/data/.tokens/laads'


def _year_from_filename(path: str) -> int:
    """Extract a 4-digit year from a raster filename."""
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
        prog='batch_fire_mapping_viirs_web',
        description='Web interface for user-defined Sentinel-2 fire mapping '
                    '(VIIRS hint, multi-year).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example
-------
  python -m batch_fire_mapping_viirs_web                            \\
      --rasters  pgfc_2022.bin  pgfc_2023.bin  pgfc_2024.bin       \\
      --out_root /data/bill/mapping_results                         \\
      --insecure_no_auth --padding 0.2
        """,
    )

    # Required
    p.add_argument('--rasters', nargs='+', required=True,
                   help='One or more Sentinel-2 ENVI .bin rasters; each '
                        'filename must contain a unique 4-digit year.')
    p.add_argument('--out_root', required=True,
                   help='Mother directory; per-year outputs go to '
                        '<out_root>/<raster_stem>_mapping_results.')
    p.add_argument('--year', type=int, default=None,
                   help='Initial active year (default: value stored in '
                        '<out_root>/active_year.yaml, else newest year).')

    # Sampling defaults
    p.add_argument('--padding', type=float, default=0.1,
                   help='Crop padding fraction (default: 0.1).')
    p.add_argument('--sample_rate', type=float, default=0.05,
                   help='Default sample rate (default: 0.05)')
    p.add_argument('--min_samples', type=int, default=500)
    p.add_argument('--max_samples', type=int, default=30000)

    # VIIRS prepare workers
    p.add_argument('--viirs_concurrent_jobs', type=int, default=1,
                   help='How many VIIRS prepare jobs run in parallel '
                        '(default: 1; FIFO queue).')
    p.add_argument('--viirs_download_workers', type=int, default=16,
                   help='Per-job parallel LAADS download workers '
                        '(default: 16).')
    p.add_argument('--viirs_shapify_workers', type=int, default=8,
                   help='Per-job parallel shapify workers (default: 8).')

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

    # Token override (mostly for tests)
    p.add_argument('--laads_token_file', default=_LAADS_TOKEN_PATH,
                   help=f'Path to LAADS DAAC token file '
                        f'(default: {_LAADS_TOKEN_PATH})')

    return p


def _load_laads_token(path: str) -> str:
    """Read the LAADS token file. Exits with an actionable message if
    missing / unreadable."""
    if not os.path.isfile(path):
        sys.exit(
            f'ERROR: LAADS DAAC token file not found at {path}.\n'
            f'  Create it with your token (one line). See:\n'
            f'  https://ladsweb.modaps.eosdis.nasa.gov/profile/#app-keys')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            tok = f.read().strip()
    except OSError as exc:
        sys.exit(f'ERROR: Cannot read LAADS token file {path}: {exc}')
    if not tok:
        sys.exit(f'ERROR: LAADS token file {path} is empty.')
    return tok


def _ensure_overviews(rasters_by_year: dict, shared_root: str):
    """Generate per-year overview PNG + sidecar JSON. Returns (png_map,
    meta_map).

    Always regenerates at server startup, regardless of the on-disk
    mtime/size cache -- this is the one moment we know for certain
    which raster is actually being served, so it's worth paying the
    (one-time, at-startup) cost of a full read to guarantee the
    overview and its reported band names can't be stale leftovers
    from a previous raster. ensure_overview()'s normal freshness
    check still applies everywhere else this cache is read (e.g.
    per-fire crop preview), so this only affects the startup cost.
    """
    from .overview import generate_overview

    cache_dir = os.path.join(shared_root, '.web_cache', '_overviews')
    os.makedirs(cache_dir, exist_ok=True)
    png_map: dict = {}
    meta_map: dict = {}
    for y in sorted(rasters_by_year):
        raster = rasters_by_year[y]
        stem = os.path.splitext(os.path.basename(raster))[0]
        png = os.path.join(cache_dir, f'{stem}.png')
        meta = os.path.join(cache_dir, f'{stem}.json')
        sys.stderr.write(
            f'[overview] Regenerating {os.path.basename(png)} from '
            f'{os.path.basename(raster)} (forced at startup) ...\n')
        sys.stderr.flush()
        generate_overview(raster, png, meta, max_dim=2000)
        png_map[y] = png
        meta_map[y] = meta
    return png_map, meta_map


def main():
    args = _build_parser().parse_args()

    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    # ------------------------------------------------------------------
    # LAADS token (fail fast)
    # ------------------------------------------------------------------
    laads_token = _load_laads_token(os.path.abspath(args.laads_token_file))

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
    for y, r in rasters_by_year.items():
        stem = os.path.splitext(os.path.basename(r))[0]
        od = os.path.join(out_root, f'{stem}_mapping_results')
        os.makedirs(od, exist_ok=True)
        outdirs_by_year[y] = od

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

    raster_path = rasters_by_year[active_year]
    output_root = outdirs_by_year[active_year]

    sep = '=' * 60
    print(f'\n{sep}')
    print('  BATCH FIRE MAPPING — VIIRS WEB INTERFACE (multi-year)')
    print(sep)
    print(f'  Out root   : {out_root}')
    print(f'  Years      : {sorted(rasters_by_year)}')
    print(f'  Active year: {active_year}')
    print(f'  Raster     : {raster_path}')
    print(f'  Output     : {output_root}')
    print(f'  LAADS token: {args.laads_token_file}')
    print(sep)

    # ------------------------------------------------------------------
    # Step 1 — Generate per-year overview PNG + sidecar JSON (cached)
    # ------------------------------------------------------------------
    print('\n[1/4] Per-year overview previews ...')
    overview_png_by_year, overview_meta_by_year = _ensure_overviews(
        rasters_by_year, out_root)

    # ------------------------------------------------------------------
    # Step 2 — Initialise application state
    # ------------------------------------------------------------------
    print('\n[2/4] Initialising AppState ...')
    crs_wkt, gt, W, H = get_raster_info(raster_path)

    from .state import AppState
    from .app import init_app, create_server

    app_state = AppState()
    app_state.raster_path    = raster_path
    app_state.raster_crs     = crs_wkt
    app_state.raster_gt      = gt
    app_state.raster_W       = W
    app_state.raster_H       = H
    app_state.output_root    = output_root

    app_state.active_year             = active_year
    app_state.shared_root             = out_root
    app_state.rasters_by_year         = rasters_by_year
    app_state.outdirs_by_year         = outdirs_by_year
    app_state.overview_png_by_year    = overview_png_by_year
    app_state.overview_meta_by_year   = overview_meta_by_year

    app_state.project_root   = _PROJECT_ROOT
    app_state.cli_script     = os.path.join(
        _REPO_ROOT, 'py', 'fire_mapping', 'fire_mapping_cli.py')
    app_state.padding        = args.padding
    app_state.sample_rate    = args.sample_rate
    app_state.min_samples    = args.min_samples
    app_state.max_samples    = args.max_samples
    app_state.laads_token    = laads_token
    app_state.viirs_concurrent_jobs = max(1, int(args.viirs_concurrent_jobs))
    app_state.viirs_download_workers = max(1, int(args.viirs_download_workers))
    app_state.viirs_shapify_workers = max(1, int(args.viirs_shapify_workers))
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

    # CSRF allowed origins
    app_state.allowed_origins = {
        f'http://localhost:{args.port}',
        f'http://127.0.0.1:{args.port}',
    }
    if args.host not in ('127.0.0.1', 'localhost'):
        app_state.allowed_origins.add(f'http://{args.host}:{args.port}')

    if not os.path.isfile(app_state.cli_script):
        sys.exit(f'ERROR: fire_mapping_cli.py not found at '
                 f'{app_state.cli_script}')

    # Load recommended settings (shared across years)
    app_state.settings_file = os.path.join(out_root,
                                           'recommended_settings.yaml')
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
            f'ERROR: {_settings_path} uses the legacy size-bucket schema.')
    if not isinstance(_cfg, dict) or 'settings' not in _cfg:
        sys.exit(
            f'ERROR: {_settings_path} missing required key "settings".')

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
          f'recommended setting(s). K={app_state.k_runs_per_setting}, '
          f'jitter={app_state.k_jitter}')

    # IP/session persistence
    app_state.ip_file = os.path.join(out_root, 'access_control.yaml')
    if os.path.isfile(app_state.ip_file):
        try:
            with open(app_state.ip_file) as _f:
                _ip_data = yaml.safe_load(_f) or {}
            app_state.approved_ips = _ip_data.get('approved', {})
            app_state.blocked_ips = _ip_data.get('blocked', {})
            app_state.pending_ips = _ip_data.get('pending', {})
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
        except Exception as _e:
            print(f'      WARNING: Failed to load sessions: {_e}')

    # ------------------------------------------------------------------
    # Step 3 — Wire up application state and start the server
    # immediately. The VIIRS bootstrap (download + shapify + index) is
    # network-bound and can retry indefinitely on failure (see
    # year_viirs.py) -- it must NOT block the server from accepting
    # connections, or an outage on NASA's end becomes an outage of
    # this app too. Everything in this step is local/disk-only and
    # fast, so it's safe to run before the listening socket opens.
    # ------------------------------------------------------------------
    from . import year_viirs
    for _y in sorted(rasters_by_year):
        app_state.viirs_shp_dirs_by_year[_y] = year_viirs.year_shp_dir(
            app_state, _y)

    app_state.init_fires_from_disk()

    init_app(app_state)

    # Restore per-fire state from previous session for the active year
    from .app import (_load_fire_state, _save_active_year,
                      _load_stage_timings, _load_notifications,
                      _load_cache_retention, _cache_sweep_loop)
    _load_fire_state()
    _save_active_year()

    _load_stage_timings()
    _load_notifications()
    _load_cache_retention()

    import threading as _threading
    _threading.Thread(target=_cache_sweep_loop,
                      daemon=True).start()

    # ------------------------------------------------------------------
    # Step 4 — Start the server
    # ------------------------------------------------------------------
    print('\n[3/4] Starting web server ...')
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
    print(f'  VIIRS bootstrap running in background -- until it')
    print(f'  completes, visitors see a brief "starting up" page.')
    print(f'{sep}\n')

    # ------------------------------------------------------------------
    # Step 5 — Bootstrap year-wide VIIRS data in the background.
    # Retries indefinitely per-day/per-step (year_viirs.retry_forever);
    # never silently skips data. state.startup_complete gates the
    # placeholder page in handlers/base.py until this finishes.
    # ------------------------------------------------------------------
    def _run_bootstrap():
        print('\n[4/4] Bootstrapping per-year VIIRS data '
              '(download + shapify) -- running in background ...')
        try:
            year_viirs.bootstrap_all_years(app_state)
        except Exception as _exc:
            # Should be unreachable -- every step inside
            # bootstrap_all_years retries forever instead of raising.
            # If something still gets here, it's a genuine bug, and we
            # still want the app usable rather than stuck, so log it
            # and let the gate open with a recorded error.
            app_state.startup_error = str(_exc)
            sys.stderr.write(
                f'      WARNING: VIIRS bootstrap raised unexpectedly: '
                f'{_exc}\n')
        if not year_viirs.shutdown_event.is_set():
            app_state.startup_complete = True
            app_state.startup_progress = {
                'stage': 'ready', 'detail': 'VIIRS bootstrap complete',
            }
            print('\n[4/4] VIIRS bootstrap complete -- full app now '
                  'available to visitors.\n')

    bootstrap_thread = _threading.Thread(
        target=_run_bootstrap, daemon=True, name='viirs-bootstrap')
    bootstrap_thread.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down...')
        year_viirs.shutdown_event.set()
        server.shutdown()


if __name__ == '__main__':
    main()
