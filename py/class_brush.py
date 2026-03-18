"""20260318
class_brush.py — Fire boundary tracing from Sentinel-2 binary mask

Usage:
  python3 class_brush.py <mask.bin> <source_scene.bin> [brush_size] [point_threshold]

Parameters (command-line):
  argv[1]  : input file — binary mask, ENVI type-4 (float32), single band
  argv[2]  : source data file — Sentinel-2 scene (used for clipping and naming)
  argv[3]  : brush_size — linking window width in pixels (optional, default 15)
  argv[4]  : point_threshold — minimum pixel count to process a component
               (optional, default 10)

Constants (edit at top of file):
  WRITE_PNG   : write debug PNG visualisations (default False)

Outputs (per detected boundary component):
  23_<ID>_<date>_<HHMM>_detection_sentinel2.kml   — polygon boundary
  23_<ID>_<date>_<HHMM>_detection_sentinel2.tif   — clipped GeoTIFF

Additional outputs for the largest (assumed fire) component:
  23_<date>_<HHMM>_detection_sentinel2.kml         — polygon boundary
  23_<date>_<HHMM>_detection_sentinel2.tif         — clipped GeoTIFF

Band selection for TIF output:
  Reads band names from the source .hdr. Finds all consecutive runs of bands
  whose names contain B12, B11, or B9. Selects the second such run (e.g.
  B12_post, B11_post, B9_post). Falls back to all bands if fewer than two
  runs are found.

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
DEFAULT_BRUSH_SIZE  = 15
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
        '[brush_size, default 15] [point_threshold, default 10]')

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

# ── parse band names from source HDR and select plotting bands ───────────────
# Reads band names from the source .hdr file. Finds consecutive runs of bands
# whose names contain B12, B11, or B9. Selects the second such run.
# Returns a space-separated string of 1-based band indices, or '' for all bands.
def select_plot_bands(hdr_path):
    try:
        with open(hdr_path, 'r') as f:
            text = f.read()
        # Extract the band names block: band names = { ... }
        start = text.find('band names')
        if start < 0:
            return ''
        brace_open = text.find('{', start)
        brace_close = text.find('}', brace_open)
        if brace_open < 0 or brace_close < 0:
            return ''
        names_raw = text[brace_open + 1 : brace_close]
        names = [n.strip() for n in names_raw.split(',') if n.strip()]

        MATCH = ('B12', 'B11', 'B9')

        def matches(name):
            return any(sub in name for sub in MATCH)

        # Build list of 1-based indices of matching bands
        matching = [i + 1 for i, n in enumerate(names) if matches(n)]

        if not matching:
            return ''

        # Split into consecutive runs (gaps > 1 start a new run)
        runs = []
        run = [matching[0]]
        for idx in matching[1:]:
            if idx == run[-1] + 1:
                run.append(idx)
            else:
                runs.append(run)
                run = [idx]
        runs.append(run)

        if len(runs) < 2:
            print('WARNING: fewer than 2 consecutive band runs found; using first run')
            chosen = runs[0]
        else:
            chosen = runs[1]   # second run, e.g. _post bands

        band_str = ' '.join(str(b) for b in chosen)
        print('Selected plot bands (1-based): %s  -> %s'
              % (band_str, [names[b - 1] for b in chosen]))
        return band_str

    except Exception as e:
        print('WARNING: band name parsing failed (%s); using all bands' % e)
        return ''

src_hdr = os.path.splitext(src_data)[0] + '.hdr'
if not exists(src_hdr):
    # Try appending .hdr directly (handles foo.bin → foo.bin.hdr convention)
    src_hdr = src_data + '.hdr'
plot_bands = select_plot_bands(src_hdr)

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

# ── collect component pixel counts from C++ output ───────────────────────────
# class_brush.exe emits one line per accepted component:
#   +component <N> <pixel_count>
components = []   # list of (N, n_px) in emission order
for line in lines:
    w = line.strip().split()
    if len(w) == 3 and w[0] == '+component':
        components.append((int(w[1]), int(w[2])))

if not components:
    print('No components found above threshold.')
    run('clean')
    sys.exit(0)

# Identify the largest component by pixel count
largest_N = max(components, key=lambda x: x[1])[0]
print('Largest component: %03d' % largest_N)

# ── per-component post-processing ────────────────────────────────────────────
ts_str = str(local_hh).zfill(2) + str(local_mm).zfill(2)

for N, n_px in components:
    f_i = str(N).zfill(3)

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

    src_clip  = f_i + '.bin'
    src_cliph = f_i + '.hdr'

    # project source data onto component extent
    run('po ' + src_data + ' ' + comp_bin + ' ' + src_clip)

    string = '23_' + f_i + '_' + ds + '_' + ts_str + '_detection_sentinel2'
    print('Output label:', string)

    run('mv ' + comp_kml + ' ' + string + '.kml')
    run('mv ' + src_clip  + ' ' + string + '.bin')
    run('mv ' + src_cliph + ' ' + string + '.hdr')

    binfile = string + '.bin'
    envi_cmd = 'envi2tif.py ' + binfile
    if plot_bands:
        envi_cmd += ' ' + plot_bands
    run(envi_cmd)
    run('mv ' + binfile + '_ht.bin_smult.tif ' + string + '.tif')

    # ── largest component: also write the no-index output pair ───────────────
    if N == largest_N:
        fire_base = '23_' + ds + '_' + ts_str + '_detection_sentinel2'
        run('cp ' + string + '.kml ' + fire_base + '.kml')
        run('cp ' + string + '.tif ' + fire_base + '.tif')
        print('Fire output:', fire_base + '.kml', fire_base + '.tif')

# ── final permissions and cleanup ────────────────────────────────────────────
run('chmod 755 23_*.tif')
run('chmod 755 23_*.kml')
run('rm -f *smult*')
run('rm -f *bin_ht*')
run('clean')
