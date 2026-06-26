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


def _ts() -> str:
    """Current timestamp as [YYYY-MM-DD HH:MM:SS], for prefixing every
    startup status/debug/update message so it's clear when (and
    whether) the server is stuck on a given step."""
    return datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')


def _log(msg: str = '') -> None:
    """print() with a timestamp prefix on every line (msg may contain
    embedded \\n -- each line gets its own timestamp so multi-line
    blocks stay readable when interleaved with other timestamped
    output)."""
    ts = _ts()
    for line in str(msg).split('\n'):
        print(f'{ts} {line}', flush=True)


def _elog(msg: str = '') -> None:
    """sys.stderr.write() with a timestamp prefix, matching _log()."""
    ts = _ts()
    for line in str(msg).rstrip('\n').split('\n'):
        sys.stderr.write(f'{ts} {line}\n')
    sys.stderr.flush()


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


    # Startup behaviour toggles
    p.add_argument("--skip_viirs_bootstrap", action="store_true",
                   help="Skip the year-wide VIIRS download step at startup.")
    p.add_argument("--disable_overview_force_regeneration",
                   action="store_true",
                   help="Skip forced overview regeneration at startup.")
    p.add_argument("--viirs_download_method",
                   choices=["curl_primary", "urllib_primary"],
                   default="curl_primary",
                   help="VIIRS download method order: curl_primary "
                        "(default) uses curl first with urllib fallback; "
                        "urllib_primary uses urllib first with curl "
                        "fallback (the original order).")
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


def _ensure_overviews(rasters_by_year: dict, shared_root: str,
                      force: bool = True):
    """Generate per-year overview PNG + sidecar JSON. Returns (png_map,
    meta_map)."""
    from .overview import generate_overview, ensure_overview
    cache_dir = os.path.join(shared_root, '.web_cache', '_overviews')
    os.makedirs(cache_dir, exist_ok=True)
    png_map: dict = {}
    meta_map: dict = {}
    for y in sorted(rasters_by_year):
        raster = rasters_by_year[y]
        stem = os.path.splitext(os.path.basename(raster))[0]
        png = os.path.join(cache_dir, f'{stem}.png')
        meta = os.path.join(cache_dir, f'{stem}.json')
        if force:
            _elog(
                f'[overview] Regenerating {os.path.basename(png)} from '
                f'{os.path.basename(raster)} (forced at startup) ...')
            generate_overview(raster, png, meta, max_dim=9090)
            _elog(f'[overview] Done: {os.path.basename(png)}')
        else:
            ensure_overview(raster, png, meta, max_dim=9090)
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
    _log(f'\n{sep}')
    _log('  BATCH FIRE MAPPING — VIIRS WEB INTERFACE (multi-year)')
    _log(sep)
    _log(f'  Out root   : {out_root}')
    _log(f'  Years      : {sorted(rasters_by_year)}')
    _log(f'  Active year: {active_year}')
    _log(f'  Raster     : {raster_path}')
    _log(f'  Output     : {output_root}')
    _log(f'  LAADS token: {args.laads_token_file}')
    _log(sep)

    # ------------------------------------------------------------------
    # Step 1 — Generate per-year overview PNG + sidecar JSON (cached)
    # ------------------------------------------------------------------
    _log('\n[1/4] Per-year overview previews: starting ...')
    overview_png_by_year, overview_meta_by_year = _ensure_overviews(
        rasters_by_year, out_root,
        force=not args.disable_overview_force_regeneration)
    _log('[1/4] Per-year overview previews: done.')

    # ------------------------------------------------------------------
    # Step 2 — Initialise application state
    # ------------------------------------------------------------------
    _log('\n[2/4] Initialising AppState: starting ...')
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

    _log(f'      Loaded {len(app_state.recommended_settings)} '
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
            _log(f'      WARNING: Failed to load IP list: {_e}')

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
            _log(f'      WARNING: Failed to load sessions: {_e}')

    _log('[2/4] Initialising AppState: done.')

    # ------------------------------------------------------------------
    # Step 3 — Bootstrap year-wide VIIRS data (download + shapify once).
    # Per-fire prepare then only has to ``accumulate`` from this shared
    # dir — no per-fire LAADS calls and no per-fire shapify.
    # ------------------------------------------------------------------
    from . import year_viirs

    _log('\n[3/4] VIIRS data migration from previous stack dates: '
         'starting ...')
    _migration = year_viirs.migrate_stale_viirs_data(
        out_root, set(outdirs_by_year.values()))
    if _migration['moved'] or _migration['overwritten']:
        _log(f"      Moved {_migration['moved']} .nc file(s) from "
             f"previous stack folder(s) into the active one "
             f"({_migration['overwritten']} overwrote an existing "
             f"file with the same name, "
              f"{_migration['overwritten_mismatched']} of those had "
              f"DIFFERING content -- see warnings above if so).")
    else:
        _log('      Nothing to recover (no previous stack folders, or '
             'nothing in them).')
    for _err in _migration['errors']:
        _elog(f'      WARNING: VIIRS migration: {_err}')
    _log('[3/4] VIIRS data migration from previous stack dates: done.')

    _log('\n[3/4] Purging existing VIIRS shapefiles for the active '
         'stack: starting (so they get fully regenerated from all '
         '.nc files, including anything just migrated or about to '
         'be downloaded) ...')
    _purged = year_viirs.purge_active_shapefiles(
        set(outdirs_by_year.values()))
    _log(f'      Removed {_purged} shapefile component(s).')
    _log('[3/4] Purging existing VIIRS shapefiles: done.')

    for _y in sorted(rasters_by_year):
        app_state.viirs_shp_dirs_by_year[_y] = year_viirs.year_shp_dir(
            app_state, _y)

    if not args.skip_viirs_bootstrap:
        _log('\n[3/4] LAADS DAAC credentials/connectivity check: '
             'starting ...')
        _preflight_log_dir = year_viirs.year_viirs_dir(
            app_state, app_state.active_year)
        _preflight = year_viirs.check_laads_credentials(
            laads_token, log_dir=_preflight_log_dir)
        _status_label = {
            'ok': 'OK',
            'bad_token': 'BAD TOKEN',
            'http_error': 'SERVER ERROR',
            'unreachable': 'UNREACHABLE',
            'unknown': 'UNKNOWN',
        }.get(_preflight['status'], _preflight['status'].upper())
        _log(f"      LAADS preflight: {_status_label} "
             f"-- {_preflight['detail']}")
        if _preflight['status'] != 'ok':
            _elog(
                f"      WARNING: LAADS preflight check did not pass "
                f"cleanly ({_preflight['status']}). The bootstrap "
                f"below may fail or download nothing for this reason "
                f"-- see the line above for which case this is "
                f"(bad token vs. server/network issue).")
        _log('[3/4] LAADS DAAC credentials/connectivity check: done.')
    if args.skip_viirs_bootstrap:
        _log('\n[3/4] Skipping VIIRS bootstrap (--skip_viirs_bootstrap). '
             'Per-fire creation will fall back to on-demand download.')
    else:
        _curl_primary = (args.viirs_download_method == 'curl_primary')
        _method_label = ('curl primary, urllib fallback'
                         if _curl_primary
                         else 'urllib primary, curl fallback')
        _log(f'\n[3/4] Bootstrapping per-year VIIRS data '
             f'(download + shapify, {_method_label}): starting ...')
        try:
            year_viirs.bootstrap_all_years(app_state,
                                           curl_primary=_curl_primary)
            _log('[3/4] Bootstrapping per-year VIIRS data: done.')
        except Exception as _exc:
            _elog(
                f'      WARNING: VIIRS bootstrap failed: {_exc}\n'
                f'      Per-fire creation will fall back to on-demand '
                f'download.')
            _log('[3/4] Bootstrapping per-year VIIRS data: FAILED.')

    app_state.init_fires_from_disk()

    init_app(app_state)

    # Download the latest BCWS current-fire points + polygons once at
    # startup, so the overlay is already populated when /new_fire is
    # first opened (rather than only after someone clicks the manual
    # refresh button). Non-fatal: data.gov.bc.ca being unreachable at
    # boot shouldn't prevent the server from starting -- the overlay
    # just stays empty until the button is used.
    _log('\n[bcws] Downloading current-fire points + polygons: '
         'starting ...')
    try:
        from . import bcws
        _overlay = bcws.refresh_bcws_overlay(app_state)
        _log(f"      {_overlay['n_points']} point(s), "
             f"{_overlay['n_polygons']} polygon(s) downloaded.")
        _log('[bcws] Downloading current-fire points + polygons: done.')
    except Exception as _exc:
        _elog(
            f'      WARNING: BCWS download failed: {_exc}\n'
            f'      Points/polygons overlay will be empty until the '
            f'"Update BCWS points + polys" button is used.')
        _log('[bcws] Downloading current-fire points + polygons: FAILED.')

    # Restore per-fire state from previous session for the active year
    _log('\n[startup] Restoring per-fire state from previous session: '
         'starting ...')
    from .app import (_load_fire_state, _save_active_year,
                      _load_stage_timings, _load_notifications,
                      _load_cache_retention, _cache_sweep_loop)
    _load_fire_state()
    _save_active_year()

    _load_stage_timings()
    _load_notifications()
    _load_cache_retention()
    _log('[startup] Restoring per-fire state from previous session: done.')

    import threading as _threading
    _threading.Thread(target=_cache_sweep_loop,
                      daemon=True).start()

    # ------------------------------------------------------------------
    # Step 4 — Start the server
    # ------------------------------------------------------------------
    _log('\n[4/4] Starting web server: starting ...')
    server = create_server(args.host, args.port)
    _log('[4/4] Starting web server: done.')

    _log(f'\n{sep}')
    _log(f'  Server ready!')
    _log(f'  Local:   http://localhost:{args.port}')
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        lan_ip = s.getsockname()[0]
        s.close()
        hostname = socket.gethostname()
        _log(f'  Network: http://{lan_ip}:{args.port}')
        _log(f'           http://{hostname}:{args.port}')
        app_state.allowed_origins.add(f'http://{lan_ip}:{args.port}')
        app_state.allowed_origins.add(f'http://{hostname}:{args.port}')
    except Exception:
        pass
    if app_state.admin_password:
        _log(f'  Auth:    admin + user passwords configured')
        _log(f'  IP ctrl: {app_state.ip_file}')
    else:
        _log(f'  Auth:    NONE (--insecure_no_auth)')
        _log(f'  WARNING: All users have full admin access!')
    _log(f'  Years:   {sorted(app_state.rasters_by_year)} '
         f'(active={app_state.active_year})')
    _log(f'  {len(app_state.fires)} fire(s) available')
    _log(f'{sep}\n')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log('\nShutting down...')
        server.shutdown()


if __name__ == '__main__':
    main()
