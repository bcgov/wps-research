"""20260318
class_brush.py — Fire boundary tracing from Sentinel-2 binary mask

Usage:
  python3 class_brush.py <mask.bin> <source_scene.bin> [brush_size] [point_threshold]

Parameters (command-line):
  argv[1]  : input file — binary mask, ENVI type-4 (float32), single band
  argv[2]  : source data file — Sentinel-2 scene (used for clipping and naming)
  argv[3]  : brush_size — linking window width in pixels (optional, default 111)
  argv[4]  : point_threshold — minimum pixel count to process a component
               (optional, default 10)

Constants (edit at top of file):
  WRITE_PNG   : write debug PNG visualisations (default False)

Outputs (per detected boundary component):
  23_<ID>_<date>_<HHMM>_detection_sentinel2.kml   — polygon boundary
  23_<ID>_<date>_<HHMM>_detection_sentinel2.tif   — clipped GeoTIFF

Dependencies:
  class_brush      (compiled from class_brush.cpp, must be on PATH or in ../cpp/)
  binary_polygonize.py  (appends .kml to its input argument, e.g. foo.bin → foo.bin.kml)
  raster_plot.py
  envi2tif.py
  po               (project/clip utility)
  timezonefinder   (pip install timezonefinder)
  pyproj           (pip install pyproj)
"""

WRITE_PNG = False

# ── defaults (used when optional CLI args are absent) ────────────────────────
DEFAULT_BRUSH_SIZE  = 111
DEFAULT_POINT_THRES = 10

import os
import sys
from datetime import timezone
from misc import err, run, pd, sep, exists, args

# ── locate compiled tools ────────────────────────────────────────────────────
cd = pd + '..' + sep + 'cpp' + sep
BRUSH_EXE = cd + 'class_brush.exe'

# ── argument validation ──────────────────────────────────────────────────────
if len(args) < 3:
    err('class_brush.py [input mask .bin] [source scene .bin] '
        '[brush_size, default 111] [point_threshold, default 10]')

fn, src_data = args[1], args[2]

if not exists(fn):
    err('Please check input file: ' + fn)
if not exists(src_data):
    err('Please check input file: ' + src_data)

# ── optional numeric parameters ──────────────────────────────────────────────
try:
    BRUSH_SIZE = int(args[3]) if len(args) >= 4 else DEFAULT_BRUSH_SIZE
except ValueError:
    err('brush_size must be an integer, got: ' + args[3])

try:
    POINT_THRES = int(args[4]) if len(args) >= 5 else DEFAULT_POINT_THRES
except ValueError:
    err('point_threshold must be an integer, got: ' + args[4])

if BRUSH_SIZE <= 0:
    err('brush_size must be > 0, got: ' + str(BRUSH_SIZE))
if POINT_THRES <= 0:
    err('point_threshold must be > 0, got: ' + str(POINT_THRES))

print('brush_size=%d  point_threshold=%d' % (BRUSH_SIZE, POINT_THRES))

# ── parse timestamp from Sentinel-2 filename ────────────────────────────────
# Expected pattern: ..._YYYYMMDDTHHMMSS_...
w = src_data.split('_')
ts_raw = None
ds = None
for token in w:
    if len(token) == 15 and 'T' in token:
        parts = token.split('T')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            ds = parts[0]       # YYYYMMDD
            ts_raw = parts[1]   # HHMMSS
            break

if ds is None or ts_raw is None:
    err('Could not parse date/time from source filename: ' + src_data)

utc_hh = int(ts_raw[0:2])
utc_mm = int(ts_raw[2:4])

# ── resolve local time offset ────────────────────────────────────────────────
# Attempt to read raster coordinates from the ENVI header for a precise TZ
# lookup via timezonefinder. Falls back to UTC-7 (PDT) if unavailable.
local_offset_hours = -7   # default fallback: PDT

try:
    from timezonefinder import TimezoneFinder
    from misc import hread_coords  # returns (centre_lat, centre_lon) from header

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

# ── run the combined C++ processing pipeline ────────────────────────────────
cmd = (
    BRUSH_EXE
    + ' ' + fn
    + ' ' + str(BRUSH_SIZE)
    + ' ' + str(POINT_THRES)
)
print('Running:', cmd)
lines = os.popen(cmd).read().split('\n')
for line in lines:
    print(line)

# ── parse per-component output lines and post-process ───────────────────────
# class_brush.exe emits one line per accepted component:
#   +component <N> <pixel_count>
for line in lines:
    w = line.strip().split()
    if len(w) != 3 or w[0] != '+component':
        continue

    N    = int(w[1])
    n_px = int(w[2])
    f_i  = str(N).zfill(3)

    # Binary mask written by class_brush.exe for this component
    comp_bin = fn + '_comp_' + f_i + '.bin'

    if not exists(comp_bin):
        print('WARNING: expected component file not found:', comp_bin)
        continue

    # ── polygonize ───────────────────────────────────────────────────────────
    # binary_polygonize.py appends .kml directly to its input path, so the
    # output is <comp_bin>.kml (e.g. fire_mapping_result.bin_comp_001.bin.kml).
    # We call via os.system() so that a non-zero exit code (e.g. from GDAL
    # deprecation warnings) does not abort the loop. Success is confirmed by
    # checking the output file exists on disk.
    poly_cmd = 'python3 ' + pd + 'binary_polygonize.py ' + comp_bin
    print('run:', poly_cmd)
    os.system(poly_cmd)

    comp_kml = comp_bin + '.kml'   # matches what binary_polygonize.py writes
    if not exists(comp_kml):
        print('WARNING: polygonize did not produce expected KML:', comp_kml)
        print('Skipping component', N)
        continue

    if WRITE_PNG:
        run(['python3', pd + 'raster_plot.py', comp_bin, '1 2 3 1 1 &'])

    FIRE_NUM  = f_i
    src_clip  = f_i + '.bin'
    src_cliph = f_i + '.hdr'

    # project source data onto component extent
    run('po ' + src_data + ' ' + comp_bin + ' ' + src_clip)

    string = (
        '23_' + FIRE_NUM
        + '_' + ds
        + '_' + str(local_hh).zfill(2) + str(local_mm).zfill(2)
        + '_detection_sentinel2'
    )
    print('Output label:', string)

    run('mv ' + comp_kml + ' ' + string + '.kml')
    run('mv ' + src_clip  + ' ' + string + '.bin')
    run('mv ' + src_cliph + ' ' + string + '.hdr')

    binfile = string + '.bin'
    run('envi2tif.py ' + binfile)
    run('mv ' + binfile + '_ht.bin_smult.tif ' + string + '.tif')

# ── final permissions and cleanup ────────────────────────────────────────────
run('chmod 755 23_*.tif')
run('chmod 755 23_*.kml')
run('rm -f *smult*')
run('rm -f *bin_ht*')
run('clean')
