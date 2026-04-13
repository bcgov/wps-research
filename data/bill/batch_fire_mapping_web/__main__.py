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
    p.add_argument('--padding', type=float, default=0.2,
                   help='Default crop padding (default: 0.2 = 20%%)')

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
                   help='Server bind address (default: 0.0.0.0)')
    p.add_argument('--port', type=int, default=8765,
                   help='Server port (default: 8765)')

    # Authentication (env vars FIRE_AUTH_USER / FIRE_AUTH_PASSWORD also work)
    p.add_argument('--user', default=None,
                   help='Username for HTTP Basic Auth (or env FIRE_AUTH_USER)')
    p.add_argument('--password', default=None,
                   help='Password for HTTP Basic Auth (or env FIRE_AUTH_PASSWORD)')

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
    print(f'  Padding    : {args.padding * 100:.0f}%')
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
    app_state.padding        = args.padding
    app_state.sample_rate    = args.sample_rate
    app_state.min_samples    = args.min_samples
    app_state.max_samples    = args.max_samples
    app_state.perimeter_mode = args.perimeter_mode
    app_state.auth_user      = args.user or os.environ.get('FIRE_AUTH_USER')
    app_state.auth_password  = args.password or os.environ.get('FIRE_AUTH_PASSWORD')

    if not os.path.isfile(app_state.cli_script):
        sys.exit(f'ERROR: fire_mapping_cli.py not found at '
                 f'{app_state.cli_script}')

    app_state.init_fires_from_gdf()

    init_app(app_state)

    server = create_server(args.host, args.port)

    print(f'\n{sep}')
    print(f'  Server ready!')
    print(f'  Local:   http://localhost:{args.port}')
    import socket
    try:
        # Get actual LAN IP by connecting to an external address
        # (no data is sent — just determines which interface would be used)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        lan_ip = s.getsockname()[0]
        s.close()
        hostname = socket.gethostname()
        print(f'  Network: http://{lan_ip}:{args.port}')
        print(f'           http://{hostname}:{args.port}')
    except Exception:
        pass
    if args.user:
        print(f'  Auth:    user={args.user}')
    else:
        print(f'  Auth:    none (use --user/--password to enable)')
    print(f'  {len(app_state.fires)} fire(s) available')
    print(f'{sep}\n')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down...')
        server.shutdown()


if __name__ == '__main__':
    main()
