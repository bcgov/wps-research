"""
class_brush.py — Fire boundary tracing from Sentinel-2 binary mask

Usage:
  python3 class_brush.py <mask.bin> <source_scene.bin> [brush_size] [point_threshold] [--all_segments] [--debug]

Parameters (command-line):
  argv[1]        : input file — binary mask, ENVI type-4 (float32), single band
  argv[2]        : source data file — Sentinel-2 scene (used for clipping and naming)
  argv[3]        : brush_size — linking window width in pixels (optional, default 15)
  argv[4]        : point_threshold — minimum pixel count to process a component
                   (optional, default 10)
  --all_segments : write all components above threshold, not just the largest
  --debug        : retain all intermediate files on disk; without this flag,
                   intermediaries are written to /dev/shm (RAM) and removed

Output filename prefix is derived from the year in the source scene timestamp
(e.g. 25_ for 2025 data).

Outputs — final fire pair (largest component, no segment index):
  YY_<date>_<HHMM>_detection_sentinel2.kml
  YY_<date>_<HHMM>_detection_sentinel2.tif

Additional per-segment outputs (only when --all_segments):
  YY_<NNN>_<date>_<HHMM>_detection_sentinel2.kml / .tif

Band selection for TIF output:
  Reads band names from the source .hdr. Finds consecutive runs of bands
  whose names contain B12, B11, or B9. Selects the second such run (e.g.
  B12_post, B11_post, B9_post). Falls back to all bands if fewer than two
  runs are found.

Dependencies:
  class_brush          (compiled from class_brush.cpp, on PATH or in ../cpp/)
  binary_polygonize.py (appends .kml to its input argument)
  raster_plot.py
  po                   (project/clip utility)
  htrim2.exe
  raster_smult.exe
  envi_header_copy_mapinfo.py
  envi_update_band_names.py
  gdal_translate       (GDAL CLI)
  timezonefinder       (pip install timezonefinder)
"""

WRITE_PNG = False

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_BRUSH_SIZE  = 15
DEFAULT_POINT_THRES = 10

import os
import sys
import uuid
import shutil
from datetime import timezone
from misc import err, run, pd, sep, exists, args

# ── locate compiled tools ─────────────────────────────────────────────────────
cd = pd + '..' + sep + 'cpp' + sep
BRUSH_EXE = cd + 'class_brush.exe'

# ── argument validation ───────────────────────────────────────────────────────
if len(args) < 3:
    err('class_brush.py [input mask .bin] [source scene .bin] '
        '[brush_size] [point_threshold] [--all_segments] [--debug]')

fn, src_data = args[1], args[2]

if not exists(fn):
    err('Please check input file: ' + fn)
if not exists(src_data):
    err('Please check input file: ' + src_data)

# Numeric positional args — only parse if the token looks like an integer
def _try_int(token, name):
    try:
        return int(token)
    except ValueError:
        err('%s must be an integer, got: %s' % (name, token))

numeric_args = [a for a in args[3:] if not a.startswith('--')]
BRUSH_SIZE  = _try_int(numeric_args[0], 'brush_size')  if len(numeric_args) >= 1 else DEFAULT_BRUSH_SIZE
POINT_THRES = _try_int(numeric_args[1], 'point_threshold') if len(numeric_args) >= 2 else DEFAULT_POINT_THRES

if BRUSH_SIZE <= 0:
    err('brush_size must be > 0, got: ' + str(BRUSH_SIZE))
if POINT_THRES <= 0:
    err('point_threshold must be > 0, got: ' + str(POINT_THRES))

ALL_SEGMENTS = '--all_segments' in args
DEBUG        = '--debug'        in args

print('brush_size=%d  point_threshold=%d  all_segments=%s  debug=%s'
      % (BRUSH_SIZE, POINT_THRES, ALL_SEGMENTS, DEBUG))

# ── RAM workspace: /dev/shm unless --debug ────────────────────────────────────
# Intermediary files from envi2tif processing go here.
TMPDIR = '/dev/shm' if (not DEBUG and os.path.isdir('/dev/shm')) else '.'

# ── parse timestamp from Sentinel-2 filename ─────────────────────────────────
# Expected pattern: ..._YYYYMMDDTHHMMSS_...
_w = src_data.split('_')
ts_raw = None
ds = None
for token in _w:
    if len(token) == 15 and 'T' in token:
        parts = token.split('T')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            ds = parts[0]       # YYYYMMDD
            ts_raw = parts[1]   # HHMMSS
            break

if ds is None or ts_raw is None:
    err('Could not parse date/time from source filename: ' + src_data)

# Output prefix derived from year in the scene timestamp (e.g. "25" for 2025)
yr = ds[2:4]

utc_hh = int(ts_raw[0:2])
utc_mm = int(ts_raw[2:4])

# ── resolve local time offset ─────────────────────────────────────────────────
local_offset_hours = -7   # fallback: PDT

try:
    from timezonefinder import TimezoneFinder
    from misc import hread_coords
    lat, lon = hread_coords(fn + '.hdr')
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon)
    if tz_name:
        import zoneinfo
        from datetime import datetime
        tz = zoneinfo.ZoneInfo(tz_name)
        dt_utc = datetime(int(ds[:4]), int(ds[4:6]), int(ds[6:8]),
                          utc_hh, utc_mm, tzinfo=timezone.utc)
        local_offset_hours = int(
            dt_utc.astimezone(tz).utcoffset().total_seconds() // 3600
        )
        print('Timezone: %s  UTC offset: %+d h' % (tz_name, local_offset_hours))
except Exception:
    print('Timezone lookup unavailable, using UTC%+d fallback.' % local_offset_hours)

local_hh = utc_hh + local_offset_hours
local_mm = utc_mm
ts_str   = str(local_hh).zfill(2) + str(local_mm).zfill(2)

# ── band selection ────────────────────────────────────────────────────────────
def select_plot_bands(hdr_path):
    """Return list of 1-based band indices (second consecutive B12/B11/B9 run),
    or empty list to use all bands."""
    try:
        with open(hdr_path, 'r') as f:
            text = f.read()
        start      = text.find('band names')
        if start < 0: return []
        brace_open  = text.find('{', start)
        brace_close = text.find('}', brace_open)
        if brace_open < 0 or brace_close < 0: return []
        names = [n.strip() for n in
                 text[brace_open + 1 : brace_close].split(',') if n.strip()]
        MATCH = ('B12', 'B11', 'B9')
        matching = [i + 1 for i, n in enumerate(names)
                    if any(sub in n for sub in MATCH)]
        if not matching: return []
        runs, cur = [], [matching[0]]
        for idx in matching[1:]:
            if idx == cur[-1] + 1:
                cur.append(idx)
            else:
                runs.append(cur); cur = [idx]
        runs.append(cur)
        if len(runs) < 2:
            print('WARNING: fewer than 2 band runs found; using first run')
            chosen = runs[0]
        else:
            chosen = runs[1]
        print('Selected plot bands (1-based): %s  -> %s'
              % (' '.join(str(b) for b in chosen), [names[b-1] for b in chosen]))
        return chosen
    except Exception as e:
        print('WARNING: band name parsing failed (%s); using all bands' % e)
        return []

src_hdr = os.path.splitext(src_data)[0] + '.hdr'
if not exists(src_hdr):
    src_hdr = src_data + '.hdr'
plot_bands = select_plot_bands(src_hdr)   # list of ints, or []

# ── inline envi2tif ───────────────────────────────────────────────────────────
def envi_to_tif(fn_bin, out_tif, band_select, src_hdr_for_meta):
    """Histogram-trim, scale to byte, and export as GeoTIFF.

    Replicates envi2tif.py entirely in-process. Intermediary files go to
    TMPDIR (/dev/shm when --debug is not set, '.' otherwise).
    band_select: list of 1-based ints, or [] for all bands.

    Pipeline:
      1. [optional] gdal_translate -of ENVI  → extract selected bands → tmp_sel
      2. htrim2.exe                           → tmp_sel_ht.bin
      3. raster_smult.exe * 255              → tmp_sel_ht_smult.bin
      4. envi_header_copy_mapinfo.py         → fix map info in smult header
      5. envi_update_band_names.py           → fix band names in smult header
      6. gdal_translate -of GTiff -ot Byte   → out_tif
      7. cleanup all tmp files
    """
    tag    = str(uuid.uuid4())[:8]
    tmp    = os.path.join(TMPDIR, 'cbrush_' + tag)
    tmp_sel        = tmp + '_sel.bin'
    tmp_sel_hdr    = tmp + '_sel.hdr'
    tmp_ht         = tmp + '_sel_ht.bin'
    tmp_ht_hdr     = tmp + '_sel_ht.hdr'
    tmp_smult      = tmp + '_sel_ht_smult.bin'
    tmp_smult_hdr  = tmp + '_sel_ht_smult.hdr'

    try:
        # Step 1: band extraction (writes ENVI pair to TMPDIR)
        if band_select:
            b_flags = ' '.join('-b ' + str(b) for b in band_select)
            run('gdal_translate -of ENVI ' + b_flags + ' ' + fn_bin + ' ' + tmp_sel)
            work = tmp_sel
            work_hdr = tmp_sel_hdr
        else:
            # No band selection — symlink or just point work at the original
            # (htrim2 will write its output alongside work, so we need a path
            #  in TMPDIR to avoid cluttering the working directory)
            run('gdal_translate -of ENVI ' + fn_bin + ' ' + tmp_sel)
            work = tmp_sel
            work_hdr = tmp_sel_hdr

        # Step 2: histogram trim (produces work + '_ht.bin' / '_ht.hdr')
        run('htrim2.exe ' + work + ' 1. 1.')

        # Step 3: scale to [0,255]
        run('raster_smult.exe ' + work + '_ht.bin 255.')

        # Step 4 & 5: copy map info and band names from the original header
        run('python3 ' + pd + 'envi_header_copy_mapinfo.py '
            + src_hdr_for_meta + ' ' + work + '_ht.bin_smult.hdr')
        run('python3 ' + pd + 'envi_update_band_names.py '
            + src_hdr_for_meta + ' ' + work + '_ht.bin_smult.hdr')

        # Step 6: final GeoTIFF
        run('gdal_translate -of GTiff -ot Byte '
            + work + '_ht.bin_smult.bin ' + out_tif)

    finally:
        # Always clean up all tmp files regardless of success/failure
        for f in [tmp_sel, tmp_sel_hdr,
                  tmp + '_sel_ht.bin',  tmp + '_sel_ht.hdr',
                  tmp + '_sel_ht.bin_smult.bin', tmp + '_sel_ht.bin_smult.hdr']:
            if os.path.exists(f):
                os.remove(f)

# ── run the C++ processing pipeline ──────────────────────────────────────────
cmd = (BRUSH_EXE
       + (' --all_segments' if ALL_SEGMENTS else '')
       + ' ' + fn
       + ' ' + str(BRUSH_SIZE)
       + ' ' + str(POINT_THRES))
print('Running:', cmd)
lines = os.popen(cmd).read().split('\n')
for line in lines:
    print(line)

# ── collect component pixel counts ───────────────────────────────────────────
components = []
for line in lines:
    _w = line.strip().split()
    if len(_w) == 3 and _w[0] == '+component':
        components.append((int(_w[1]), int(_w[2])))

if not components:
    print('No components found above threshold.')
    sys.exit(0)

largest_N = max(components, key=lambda x: x[1])[0]
print('Largest component: %03d' % largest_N)

# ── per-component post-processing ────────────────────────────────────────────
for N, n_px in components:
    f_i      = str(N).zfill(3)
    comp_bin = fn + '_comp_' + f_i + '.bin'

    if not exists(comp_bin):
        print('WARNING: expected component file not found:', comp_bin)
        continue

    # polygonize — called via os.system so non-zero exit does not abort loop
    poly_cmd = 'python3 ' + pd + 'binary_polygonize.py ' + comp_bin
    print('run:', poly_cmd)
    os.system(poly_cmd)

    comp_kml = comp_bin + '.kml'
    if not exists(comp_kml):
        print('WARNING: polygonize did not produce KML:', comp_kml, '— skipping')
        continue

    if WRITE_PNG:
        run(['python3', pd + 'raster_plot.py', comp_bin, '1 2 3 1 1 &'])

    # project source data onto component extent
    src_clip  = f_i + '.bin'
    src_cliph = f_i + '.hdr'
    run('po ' + src_data + ' ' + comp_bin + ' ' + src_clip)

    string  = yr + '_' + f_i + '_' + ds + '_' + ts_str + '_detection_sentinel2'
    binfile = string + '.bin'
    print('Output label:', string)

    run('mv ' + comp_kml + ' ' + string + '.kml')
    run('mv ' + src_clip  + ' ' + binfile)
    run('mv ' + src_cliph + ' ' + string + '.hdr')

    # convert to GeoTIFF inline (intermediaries in TMPDIR / RAM)
    envi_to_tif(binfile, string + '.tif', plot_bands, string + '.hdr')

    # ── largest component → no-index fire output pair ────────────────────────
    if N == largest_N:
        fire_base = yr + '_' + ds + '_' + ts_str + '_detection_sentinel2'
        run('cp ' + string + '.kml ' + fire_base + '.kml')
        run('cp ' + string + '.tif ' + fire_base + '.tif')
        print('Fire output:', fire_base + '.kml', fire_base + '.tif')

# ── permissions ───────────────────────────────────────────────────────────────
run('chmod 755 ' + yr + '_*.tif')
run('chmod 755 ' + yr + '_*.kml')

# ── cleanup — unless --debug, remove everything except the final fire pair ────
if not DEBUG:
    # C++ intermediaries
    for pat in [fn + '_flood4.bin',     fn + '_flood4.hdr',
                fn + '_flood4.bin_link.bin',     fn + '_flood4.bin_link.hdr',
                fn + '_flood4.bin_link.bin_recode.bin',
                fn + '_flood4.bin_link.bin_recode.hdr',
                fn + '_flood4.bin_link.bin_recode.bin_wheel.bin',
                fn + '_flood4.bin_link.bin_recode.bin_wheel.hdr']:
        if os.path.exists(pat):
            os.remove(pat)
    # component masks
    import glob
    for pat in glob.glob(fn + '_comp_*.bin') + glob.glob(fn + '_comp_*.hdr'):
        os.remove(pat)
    # per-segment indexed outputs (keep only the no-index fire pair)
    for pat in (glob.glob(yr + '_???_*_detection_sentinel2.kml') +
                glob.glob(yr + '_???_*_detection_sentinel2.tif') +
                glob.glob(yr + '_???_*_detection_sentinel2.bin') +
                glob.glob(yr + '_???_*_detection_sentinel2.hdr')):
        os.remove(pat)
    # shapefile side-products from binary_polygonize
    for pat in (glob.glob(fn + '_comp_*.bin.shp') +
                glob.glob(fn + '_comp_*.bin.shx') +
                glob.glob(fn + '_comp_*.bin.dbf') +
                glob.glob(fn + '_comp_*.bin.prj')):
        if os.path.exists(pat):
            os.remove(pat)

os.system("rm *.bak")
