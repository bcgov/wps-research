#!/usr/bin/env python3
"""apply_skip_flags.py

Run this once, from inside the batch_fire_mapping_viirs_web folder:

    python3 apply_skip_flags.py

Adds two new flags to __main__.py:
  --skip_viirs_bootstrap
  --disable_overview_force_regeneration

Both default to OFF (current behaviour unchanged) unless passed.

Uses exact string replacement (not line numbers / not a unified diff),
so it's robust to whatever line numbers your file currently has.
Edits __main__.py in place; makes a __main__.py.bak backup first.
Safe to re-run: if the edits are already present, it says so and exits
without changing anything.
"""

import shutil
import sys

PATH = "__main__.py"

REPLACEMENTS = [
    # 1. Add the two new argparse flags, right after --laads_token_file.
    (
        """    p.add_argument('--laads_token_file', default=_LAADS_TOKEN_PATH,
                   help=f'Path to LAADS DAAC token file '
                        f'(default: {_LAADS_TOKEN_PATH})')
    return p
""",
        """    p.add_argument('--laads_token_file', default=_LAADS_TOKEN_PATH,
                   help=f'Path to LAADS DAAC token file '
                        f'(default: {_LAADS_TOKEN_PATH})')

    # Startup behaviour toggles
    p.add_argument('--skip_viirs_bootstrap', action='store_true',
                   help='Skip the year-wide VIIRS download/shapify/index '
                        'step at startup entirely. Per-fire creation '
                        'falls back to on-demand download (same fallback '
                        'used if bootstrap fails). Default: unchanged '
                        '(bootstrap runs normally).')
    p.add_argument('--disable_overview_force_regeneration',
                   action='store_true',
                   help='Skip forced overview regeneration at startup; '
                        'only (re)generate when the on-disk cache is '
                        'missing or stale (mtime/size changed) -- the '
                        'normal ensure_overview() behaviour. Default: '
                        'unchanged (always regenerates at startup).')

    return p
""",
    ),
    # 2. _ensure_overviews gains a force= parameter; honours it.
    (
        '''def _ensure_overviews(rasters_by_year: dict, shared_root: str):
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
            f'{os.path.basename(raster)} (forced at startup) ...\\n')
        sys.stderr.flush()
        generate_overview(raster, png, meta, max_dim=9090)
        png_map[y] = png
        meta_map[y] = meta
    return png_map, meta_map
''',
        '''def _ensure_overviews(rasters_by_year: dict, shared_root: str,
                      force: bool = True):
    """Generate per-year overview PNG + sidecar JSON. Returns (png_map,
    meta_map).

    When force=True (default, matches prior behaviour): always
    regenerates at server startup, regardless of the on-disk
    mtime/size cache -- this is the one moment we know for certain
    which raster is actually being served, so it's worth paying the
    (one-time, at-startup) cost of a full read to guarantee the
    overview and its reported band names can't be stale leftovers
    from a previous raster.

    When force=False (--disable_overview_force_regeneration): falls
    back to ensure_overview()'s normal freshness check (only
    regenerate if missing or the raster's mtime/size changed) -- use
    this once a stack file's overview has already been generated once
    and you don't want to pay the regeneration cost on every restart
    while iterating on something unrelated.
    """
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
            sys.stderr.write(
                f'[overview] Regenerating {os.path.basename(png)} from '
                f'{os.path.basename(raster)} (forced at startup) ...\\n')
            sys.stderr.flush()
            generate_overview(raster, png, meta, max_dim=9090)
        else:
            ensure_overview(raster, png, meta, max_dim=9090)
        png_map[y] = png
        meta_map[y] = meta
    return png_map, meta_map
''',
    ),
    # 3. Call site passes force= through.
    (
        """    overview_png_by_year, overview_meta_by_year = _ensure_overviews(
        rasters_by_year, out_root)
""",
        """    overview_png_by_year, overview_meta_by_year = _ensure_overviews(
        rasters_by_year, out_root,
        force=not args.disable_overview_force_regeneration)
""",
    ),
    # 4. VIIRS bootstrap respects --skip_viirs_bootstrap.
    (
        """    print('\\n[3/4] Bootstrapping per-year VIIRS data '
          '(download + shapify) ...')
    from . import year_viirs
    for _y in sorted(rasters_by_year):
        app_state.viirs_shp_dirs_by_year[_y] = year_viirs.year_shp_dir(
            app_state, _y)
    try:
        year_viirs.bootstrap_all_years(app_state)
    except Exception as _exc:
        sys.stderr.write(
            f'      WARNING: VIIRS bootstrap failed: {_exc}\\n'
            f'      Per-fire creation will fall back to on-demand '
            f'download.\\n')
""",
        """    from . import year_viirs
    for _y in sorted(rasters_by_year):
        app_state.viirs_shp_dirs_by_year[_y] = year_viirs.year_shp_dir(
            app_state, _y)
    if args.skip_viirs_bootstrap:
        print('\\n[3/4] Skipping VIIRS bootstrap (--skip_viirs_bootstrap). '
              'Per-fire creation will fall back to on-demand download.')
    else:
        print('\\n[3/4] Bootstrapping per-year VIIRS data '
              '(download + shapify) ...')
        try:
            year_viirs.bootstrap_all_years(app_state)
        except Exception as _exc:
            sys.stderr.write(
                f'      WARNING: VIIRS bootstrap failed: {_exc}\\n'
                f'      Per-fire creation will fall back to on-demand '
                f'download.\\n')
""",
    ),
]


def main():
    with open(PATH, encoding="utf-8") as f:
        content = f.read()

    shutil.copyfile(PATH, PATH + ".bak")

    changed = 0
    skipped = 0
    for i, (old, new) in enumerate(REPLACEMENTS, start=1):
        if new in content:
            print(f"[{i}/4] already applied, skipping")
            skipped += 1
            continue
        count = content.count(old)
        if count == 0:
            print(f"[{i}/4] ERROR: expected text not found -- "
                  f"file may differ from what this script expects. "
                  f"No changes made for this block.")
            continue
        if count > 1:
            print(f"[{i}/4] ERROR: expected text found {count} times "
                  f"(expected exactly 1) -- skipping to avoid guessing "
                  f"which one to change.")
            continue
        content = content.replace(old, new, 1)
        changed += 1
        print(f"[{i}/4] applied")

    with open(PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nDone: {changed} change(s) applied, {skipped} already present.")
    print(f"Backup saved as {PATH}.bak")
    if changed == 0 and skipped < len(REPLACEMENTS):
        print("\nWARNING: some blocks were neither applied nor already "
              "present -- check the ERROR lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
