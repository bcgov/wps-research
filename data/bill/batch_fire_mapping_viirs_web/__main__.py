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
import json
import time

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
    p.add_argument("--viirs_min_interval_minutes", type=int, default=60,
                   help="Attempt the year-wide VIIRS download at most once "
                        "per this many minutes, measured across server "
                        "restarts via a stamp file in "
                        "<out_root>/.web_cache/. Set 0 to disable the "
                        "throttle and attempt on every start. Default: 60.")
    p.add_argument("--enable_viirs_download", action="store_true",
                   help="Re-enable downloading of NEW VIIRS granules. "
                        "Downloading is currently DISABLED by default "
                        "(year_viirs.VIIRS_DOWNLOAD_ENABLED): LAADS has "
                        "been slow enough that even AOI-scoped fetches "
                        "stalled fire creation for minutes. Searching, "
                        "shapifying, indexing and accumulating the .nc "
                        "files already on disk continue regardless.")
    p.add_argument("--province_wide_viirs_download", action="store_true",
                   help="Restore the old behaviour of downloading VIIRS "
                        "for the whole raster footprint at startup. Off "
                        "by default: that scan dominated boot time, and "
                        "granules are now fetched per-AOI when a fire is "
                        "confirmed. Useful for a one-off backfill.")
    p.add_argument("--force_viirs_bootstrap", action="store_true",
                   help="Ignore the --viirs_min_interval_minutes throttle "
                        "and attempt the VIIRS download on this start "
                        "regardless of when the last attempt was made.")
    p.add_argument("--disable_overview_force_regeneration",
                   action="store_true",
                   help="Skip forced overview regeneration at startup.")
    p.add_argument("--overview_min_interval_minutes", type=int, default=60,
                   help="Force-regenerate the per-year overview previews "
                        "at most once per this many minutes, measured "
                        "across server restarts via the same stamp file "
                        "used by the VIIRS throttle. When throttled, "
                        "overviews are still regenerated if the stack "
                        "file itself changed or a PNG is missing. Set 0 "
                        "to disable the throttle and force-regenerate on "
                        "every start. Default: 60.")
    p.add_argument("--viirs_download_method",
                   choices=["curl_primary", "urllib_primary"],
                   default="curl_primary",
                   help="VIIRS download method order: curl_primary "
                        "(default) uses curl first with urllib fallback; "
                        "urllib_primary uses urllib first with curl "
                        "fallback (the original order).")
    p.add_argument("--parallel_viirs_downloading", action="store_true",
                   help="Download VIIRS days concurrently across "
                        "--viirs_download_workers threads. Default is "
                        "serial (one day at a time, no thread pool) so "
                        "every curl/http entry/exit and request/response "
                        "message prints to stdout in clear chronological "
                        "order.")
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


# LOW_OVERVIEW_HEIGHT is the pyramid's coarse level: tall enough to be
# a usable map on its own, small enough to arrive quickly.
LOW_OVERVIEW_HEIGHT = 2000


def _ensure_overviews(rasters_by_year: dict, shared_root: str,
                      force: bool = True):
    """Generate per-year overview PNGs + sidecar JSON.

    Produces a two-level image pyramid per year:
      * ``<stem>.png``     -- full-size overview (longest edge <= 9090)
      * ``<stem>_low.png`` -- low-resolution level, 2000 px tall

    The new-fire page loads the low level first so the map becomes
    interactive quickly, then swaps in the full-size one once it has
    finished downloading in the background. Both are rendered from the
    same source at the same aspect ratio, so a single sidecar JSON
    describes both -- the client's coordinate math depends only on the
    image's rendered size, not its intrinsic resolution.

    Returns (png_map, low_png_map, meta_map)."""
    from .overview import generate_overview, ensure_overview
    cache_dir = os.path.join(shared_root, '.web_cache', '_overviews')
    os.makedirs(cache_dir, exist_ok=True)
    png_map: dict = {}
    low_png_map: dict = {}
    meta_map: dict = {}
    for y in sorted(rasters_by_year):
        raster = rasters_by_year[y]
        stem = os.path.splitext(os.path.basename(raster))[0]
        png = os.path.join(cache_dir, f'{stem}.png')
        meta = os.path.join(cache_dir, f'{stem}.json')
        low_png = os.path.join(cache_dir, f'{stem}_low.png')
        # The low level writes a sidecar too (generate_overview always
        # does); it is deliberately not registered anywhere -- the
        # full-size sidecar is the single source of truth for both.
        low_meta = os.path.join(cache_dir, f'{stem}_low.json')
        if force:
            _elog(
                f'[overview] Regenerating {os.path.basename(png)} + '
                f'{os.path.basename(low_png)} from '
                f'{os.path.basename(raster)} (forced at startup) ...')
            # One pass over the source produces both levels: the coarse
            # level is decimated from the full-size result in memory.
            # Reading the source twice would roughly double the wall
            # time, since these reads are seek-bound rather than
            # throughput-bound.
            generate_overview(raster, png, meta, max_dim=9090,
                              also_low_png=low_png,
                              low_target_height=LOW_OVERVIEW_HEIGHT)
            _elog(f'[overview] Done: {os.path.basename(png)} + '
                  f'{os.path.basename(low_png)}')
        else:
            ensure_overview(raster, png, meta, max_dim=9090,
                            also_low_png=low_png,
                            low_target_height=LOW_OVERVIEW_HEIGHT)
        png_map[y] = png
        low_png_map[y] = low_png
        meta_map[y] = meta
    return png_map, low_png_map, meta_map

# ----------------------------------------------------------------------
# VIIRS run stamp
#
# A small JSON file under <out_root>/.web_cache/ that survives restarts
# and records two independent things:
#
#   * every server start        (last_server_start_*, server_starts)
#   * every VIIRS download attempt (last_attempt_*, last_outcome,
#                                   attempts)
#
# The attempt timestamps drive the once-per-interval throttle, so
# restarting the server repeatedly while developing does not re-trigger
# a full LAADS download each time. The server-start fields are recorded
# unconditionally and are purely informational -- useful when you need
# to tell "the server came up" apart from "the server tried to
# download", which the log alone makes awkward to answer.
# ----------------------------------------------------------------------

_RUN_STAMP_NAME = 'viirs_run_stamp.json'


def _run_stamp_path(out_root: str) -> str:
    return os.path.join(out_root, '.web_cache', _RUN_STAMP_NAME)


def _iso_utc(ts: float) -> str:
    return datetime.datetime.fromtimestamp(
        ts, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _read_run_stamp(path: str) -> dict:
    """Return the stamp dict, or {} if absent/unreadable/corrupt."""
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        # A missing or corrupt stamp must never block startup -- it just
        # means "no known previous attempt", i.e. go ahead and download.
        return {}


def _write_run_stamp(path: str, payload: dict) -> None:
    """Atomically write the stamp. Failure is logged, never fatal."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write('\n')
        os.replace(tmp, path)
    except OSError as exc:
        _elog(f'      WARNING: could not write VIIRS run stamp '
              f'({path}): {exc}')


def _stamp_age_s(stamp: dict, key: str = 'last_attempt_epoch'):
    """Seconds since the timestamp at *key*, or None if unknown.

    A stamp dated in the future (system clock moved backwards, or the
    file was copied from another machine) is reported as unknown rather
    than as a huge negative age, so a bad clock can never wedge
    downloads off indefinitely."""
    try:
        last = float(stamp.get(key, 0) or 0)
    except (TypeError, ValueError):
        return None
    if last <= 0:
        return None
    age = time.time() - last
    return age if age >= 0 else None


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f'{h}h {m}m'
    if m:
        return f'{m}m {s}s'
    return f'{s}s'


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
    # Run stamp — read once, up front, so both the overview step below
    # and the VIIRS step further down consult the same record. The
    # server start is recorded unconditionally; the per-subsystem
    # timestamps are written only when that subsystem actually runs.
    # ------------------------------------------------------------------
    _stamp_path = _run_stamp_path(out_root)
    _stamp = _read_run_stamp(_stamp_path)
    # Ages must be read BEFORE the server-start fields are rewritten,
    # and before either step stamps itself.
    _attempt_age = _stamp_age_s(_stamp, 'last_attempt_epoch')
    _overview_age = _stamp_age_s(_stamp, 'last_overview_epoch')

    _now = time.time()
    _stamp['last_server_start_epoch'] = _now
    _stamp['last_server_start_utc'] = _iso_utc(_now)
    _stamp['server_starts'] = int(_stamp.get('server_starts', 0) or 0) + 1
    _write_run_stamp(_stamp_path, _stamp)

    # ------------------------------------------------------------------
    # Step 1 — Generate per-year overview PNG + sidecar JSON (cached)
    # ------------------------------------------------------------------
    _ovr_interval_s = max(0, int(args.overview_min_interval_minutes)) * 60
    _ovr_throttled = (
        _ovr_interval_s > 0
        and _overview_age is not None
        and _overview_age < _ovr_interval_s)
    # `force` re-renders every overview unconditionally. Dropping it
    # does NOT mean "skip the step": ensure_overview() still regenerates
    # anything whose sidecar cache_key no longer matches the stack file,
    # or whose PNG is missing. So a throttled start is still correct --
    # it just stops re-rendering images that are already up to date.
    _ovr_force = (not args.disable_overview_force_regeneration
                  and not _ovr_throttled)

    _log('\n[1/4] Per-year overview previews: starting ...')
    if _ovr_throttled:
        _remaining = _ovr_interval_s - _overview_age
        _log(f'      Last forced regeneration was '
             f'{_fmt_duration(_overview_age)} ago, within the '
             f'{args.overview_min_interval_minutes}-minute minimum '
             f'interval -- using cached overviews (any whose stack '
             f'file changed, or whose PNG is missing, are still '
             f'regenerated). Next forced regeneration allowed in '
             f'{_fmt_duration(_remaining)}.')
    (overview_png_by_year, overview_low_png_by_year,
     overview_meta_by_year) = _ensure_overviews(
        rasters_by_year, out_root,
        force=_ovr_force)
    if _ovr_force:
        _ovr_done = time.time()
        _stamp['last_overview_epoch'] = _ovr_done
        _stamp['last_overview_utc'] = _iso_utc(_ovr_done)
        _stamp['overview_regens'] = int(
            _stamp.get('overview_regens', 0) or 0) + 1
        _write_run_stamp(_stamp_path, _stamp)
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
    app_state.overview_low_png_by_year = overview_low_png_by_year
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
    try:
        app_state.max_aoi_fraction = float(
            _cfg.get('max_aoi_fraction', 0.10))
        app_state.max_aoi_fraction = max(
            0.01, min(1.0, app_state.max_aoi_fraction))
    except (TypeError, ValueError):
        app_state.max_aoi_fraction = 0.10

    _log(f'      Loaded {len(app_state.recommended_settings)} '
         f'recommended setting(s). K={app_state.k_runs_per_setting}, '
         f'jitter={app_state.k_jitter}, '
         f'max_aoi={app_state.max_aoi_fraction:.0%}')

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

    for _y in sorted(rasters_by_year):
        app_state.viirs_shp_dirs_by_year[_y] = year_viirs.year_shp_dir(
            app_state, _y)

    # ------------------------------------------------------------------
    # VIIRS run stamp: record this server start, then decide whether a
    # download attempt is due. The stamp lives under
    # <out_root>/.web_cache/ so the interval is honoured across
    # restarts -- restarting repeatedly while developing no longer
    # re-triggers a full LAADS download every time.
    # ------------------------------------------------------------------
    # Apply the download kill switch before anything can call into the
    # download paths. Off by default; --enable_viirs_download turns the
    # whole machinery back on with no code change.
    year_viirs.set_viirs_download_enabled(args.enable_viirs_download)
    if not args.enable_viirs_download:
        _log('\n[3/4] VIIRS downloading is DISABLED '
             '(pass --enable_viirs_download to turn it back on). '
             'Existing .nc files are still searched, shapified, '
             'indexed and accumulated; only fetching NEW granules is '
             'off. Fires without VIIRS coverage default to the '
             '"Red wins (post)" hint.')

    _interval_s = max(0, int(args.viirs_min_interval_minutes)) * 60
    # The throttle exists to stop repeated restarts hammering LAADS.
    # With province-wide downloading off, startup makes no LAADS calls
    # at all -- it only shapifies/indexes/accumulates what is already on
    # disk, which is local work that must run every boot so the
    # new-fire overlay reflects granules fetched per-AOI since the last
    # start. So the throttle now applies only when downloading is
    # actually enabled.
    _throttled = (
        args.province_wide_viirs_download
        and _interval_s > 0
        and not args.force_viirs_bootstrap
        and _attempt_age is not None
        and _attempt_age < _interval_s)

    if _stamp.get('last_attempt_utc'):
        _age_txt = (f'{_fmt_duration(_attempt_age)} ago'
                    if _attempt_age is not None else 'age unknown')
        _log(f"\n[3/4] Last VIIRS attempt: "
             f"{_stamp['last_attempt_utc']} ({_age_txt}, outcome: "
             f"{_stamp.get('last_outcome', 'unknown')}). "
             f"Server start #{_stamp['server_starts']}.")
    else:
        _log(f"\n[3/4] No previous VIIRS attempt on record. "
             f"Server start #{_stamp['server_starts']}.")

    # A migration that actually moved .nc files overrides the throttle:
    # those files still need shapifying, and skipping would leave them
    # unprocessed until the interval elapsed.
    if _throttled and _migration['moved']:
        _log(f"      Throttle overridden: migration moved "
             f"{_migration['moved']} file(s) that still need shapifying.")
        _throttled = False

    if args.skip_viirs_bootstrap:
        _log('\n[3/4] Skipping VIIRS bootstrap (--skip_viirs_bootstrap). '
             'Per-fire creation will fall back to on-demand download.')
    elif _throttled:
        _remaining = _interval_s - _attempt_age
        _log(f'\n[3/4] Skipping VIIRS bootstrap: last attempt was '
             f'{_fmt_duration(_attempt_age)} ago, within the '
             f'{args.viirs_min_interval_minutes}-minute minimum '
             f'interval. Next attempt allowed in '
             f'{_fmt_duration(_remaining)}. Existing shapefiles are '
             f'left intact; pass --force_viirs_bootstrap to override, '
             f'or --viirs_min_interval_minutes 0 to disable the '
             f'throttle.')
    else:
        # Purge only when a re-download is actually going to follow.
        # The purge exists so shapefiles get regenerated from every
        # .nc file; running it *without* the bootstrap below would
        # leave the year with no shapefiles at all.
        _log('\n[3/4] Purging existing VIIRS shapefiles for the active '
             'stack: starting (so they get fully regenerated from all '
             '.nc files, including anything just migrated or about to '
             'be downloaded) ...')
        _purged = year_viirs.purge_active_shapefiles(
            set(outdirs_by_year.values()))
        _log(f'      Removed {_purged} shapefile component(s).')
        _log('[3/4] Purging existing VIIRS shapefiles: done.')

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

        _curl_primary = (args.viirs_download_method == 'curl_primary')
        _method_label = ('curl primary, urllib fallback'
                         if _curl_primary
                         else 'urllib primary, curl fallback')
        _parallel_label = ('parallel' if args.parallel_viirs_downloading
                           else 'serial, no thread pool')
        _dl_label = ('province-wide download ENABLED '
                     '(--province_wide_viirs_download)'
                     if args.province_wide_viirs_download
                     else 'no province-wide download -- granules are '
                          'fetched per-AOI when a fire is confirmed')
        _log(f'\n[3/4] Indexing per-year VIIRS data '
             f'(shapify + index + accumulate; {_dl_label}; '
             f'{_method_label}, {_parallel_label}): starting ...')
        _outcome = 'unknown'
        try:
            year_viirs.bootstrap_all_years(
                app_state, curl_primary=_curl_primary,
                parallel_viirs_downloading=args.parallel_viirs_downloading,
                download=args.province_wide_viirs_download)
            _outcome = 'ok'
            _log('[3/4] Bootstrapping per-year VIIRS data: done.')
        except Exception as _exc:
            _outcome = f'failed: {_exc}'
            _elog(
                f'      WARNING: VIIRS bootstrap failed: {_exc}\n'
                f'      Per-fire creation will fall back to on-demand '
                f'download.')
            _log('[3/4] Bootstrapping per-year VIIRS data: FAILED.')
        finally:
            # Stamp the attempt whether or not it succeeded: the point
            # of the throttle is not to hammer LAADS, and a failed
            # attempt hammered it exactly as much as a successful one.
            # (A transient LAADS 500 therefore costs at most one
            # interval's delay, not an unthrottled retry loop.)
            _att = time.time()
            _stamp['last_attempt_epoch'] = _att
            _stamp['last_attempt_utc'] = _iso_utc(_att)
            _stamp['last_outcome'] = _outcome
            _stamp['attempts'] = int(_stamp.get('attempts', 0) or 0) + 1
            _write_run_stamp(_stamp_path, _stamp)

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
