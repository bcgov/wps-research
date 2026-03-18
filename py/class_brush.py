"""20260318 
class_brush.py — Fire boundary tracing from Sentinel-2 binary mask

Parameters (command-line):
  argv[1]  : input file — binary mask, ENVI type-4 (float32), single band
  argv[2]  : source data file — Sentinel-2 scene (used for clipping and naming)

Constants (edit at top of file):
  BRUSH_SIZE  : linking window width in pixels (default 111)
  POINT_THRES : minimum pixel count to process a component (default 10)
  WRITE_PNG   : write debug PNG visualisations (default False)

Outputs (per detected boundary component):
  23_<ID>_<date>_<HHMM>_detection_sentinel2.kml   — polygon boundary
  23_<ID>_<date>_<HHMM>_detection_sentinel2.tif   — clipped GeoTIFF

Dependencies:
  class_brush      (compiled from class_brush.cpp, must be on PATH or in ../cpp/)
  binary_polygonize.py
  raster_plot.py
  envi2tif.py
  po               (project/clip utility)
  timezonefinder   (pip install timezonefinder)
  pyproj           (pip install pyproj)

Usage:
  python3 class_brush.py <mask.bin> <source_scene.bin>
"""

BRUSH_SIZE  = 111
POINT_THRES = 10
WRITE_PNG   = False

import os
import sys
from datetime import timezone, timedelta
from misc import err, run, pd, sep, exists, args

# ── locate compiled tools ────────────────────────────────────────────────────
cd = pd + '..' + sep + 'cpp' + sep
BRUSH_EXE = cd + 'class_brush.exe'

# ── argument validation ──────────────────────────────────────────────────────
if len(args) < 3:
    err('class_brush.py [input file, binary mask envi type 4] [source data file]')

fn, src_data = args[1], args[2]
if not exists(fn):
    err('Please check input file: ' + fn)
if not exists(src_data):
    err('Please check input file: ' + src_data)

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
# Attempt to read raster coordinates from header for a precise TZ lookup.
# Falls back to a fixed UTC-7 (Pacific Daylight Time) offset if unavailable.
local_offset_hours = -7   # default fallback: PDT

try:
    from timezonefinder import TimezoneFinder
    import pyproj
    from misc import hread_coords  # optional helper — see note below

    # hread_coords() should return (centre_lat, centre_lon) from an ENVI header.
    # If this helper is not available the except block will catch it and fall back.
    lat, lon = hread_coords(fn + '.hdr')
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon)
    if tz_name:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(tz_name)
        from datetime import datetime
        dt_utc = datetime(int(ds[:4]), int(ds[4:6]), int(ds[6:8]),
                          utc_hh, utc_mm, tzinfo=timezone.utc)
        local_offset_hours = int(dt_utc.astimezone(tz).utcoffset().total_seconds() // 3600)
except Exception:
    pass   # silently use default offset

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

    N       = int(w[1])
    n_px    = int(w[2])
    f_i     = str(N).zfill(3)

    # Paths produced by class_brush.exe for this component
    comp_bin = fn + '_comp_' + f_i + '.bin'
    comp_kml = fn + '_comp_' + f_i + '.kml'

    if not exists(comp_bin):
        print('WARNING: expected component file not found:', comp_bin)
        continue

    # polygonize the component mask
    run(['python3', pd + 'binary_polygonize.py', comp_bin])

    if WRITE_PNG:
        run(['python3', pd + 'raster_plot.py', comp_bin, '1 2 3 1 1 &'])

    FIRE_NUM = f_i
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
# run('chmod 755 23_*.tif')
# run('chmod 755 23_*.kml')
run('rm -f *smult*')
run('rm -f *bin_ht*')
# run('clean')
