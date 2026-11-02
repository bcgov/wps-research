#!/usr/bin/env python3
"""
20261102 alos4_process.py

Polarimetric processing of ALOS-2 (PALSAR-2) and ALOS-4 (PALSAR-3) Level 1.1
stripmap data (CEOS format, from JAXA or EODMS) with ESA SNAP's command-line
tool gpt, SNAP 14 or newer. Quad-pol data become the 3x3 coherency matrix
(T3); dual-pol data the 2x2 covariance matrix (C2). Replaces
alos2_dualpol_process.py and alos2_quadpol_process.py (both left unchanged).

Which folders are processed
    Every folder in the present working directory that has a VOL-ALOS2* or
    VOL-ALOS4* file directly inside it. There is no recursive search. One
    exception, for ALOS-2 data from EODMS: a folder X whose data sit one level
    down in X/X/ (the layout py/shallow.py flattens) is processed in X/X/.
    Hidden folders (e.g. ".20250622.partial" left by alos4_unzip.py) are ignored.

    Skipped and reported: ScanSAR (several image files per polarisation),
    Spotlight, single-pol, and anything that is not complex (Level 1.1) data.

ALOS-4: converted to PALSAR-2 format first
    SNAP 14.0.0-14.0.2 misreads ALOS-4 (PALSAR-3) metadata (see Metadata
    checks below). So each ALOS-4 dataset is first converted to PALSAR-2
    (ALOS-2) CEOS format with a port of JAXA's sample code P3toP2snap, into
    <dataset>/ALOS2_format/, and SNAP reads that copy with its ALOS-2 reader.
    The copy is as big as the original data; --delete-intermediates removes
    it as soon as 01_CalMtxMlk is made (it is made again if step 01 ever has
    to run again). Outputs are still
    labelled ALOS4. JAXA's code covers the stripmap modes UWS, UBD, UWD, UBQ,
    FWD and HBQ; other modes are skipped. --native-alos4 reads the original
    files instead, once a SNAP release has the fix.

    JAXA's sample code P3toP2snap (author: Dr. Yunung Nina Lin):
        https://www.eorc.jaxa.jp/ALOS/jp/alos-4/tool/P3toP2snap.zip
    Thanks to JAXA for making this code publicly available.

Processing chain (all outputs go inside the dataset's own folder)
    ALOS2_format/    ALOS-4 only: the data in PALSAR-2 format (see above)
    probe            a 64 x 64 pixel subset, read only to check the metadata
                     before the heavy work starts (deleted afterwards)
    01_CalMtxMlk     one gpt graph: Read -> Calibration (complex output)
                     -> Polarimetric-Matrices (T3 or C2) -> Multilook.
                     The full-resolution matrix is never written to disk.
                     Multilooking happens on the matrix (incoherent averaging).
                     4 range looks; SNAP derives the azimuth looks per scene
                     for square ground pixels (--az-looks N fixes them).
    02_TF            Terrain-Flattening, Copernicus 30 m DEM       [--no-tf]
    03_BoxPre        3x3 box car filter (Polarimetric-Speckle-Filter)
                                                              [--no-pre-box]
    04_TC            Terrain-Correction: BC Albers (EPSG:3005), 20 m pixels,
                     grid aligned to multiples of 20 m so every date stacks
                     pixel for pixel; Copernicus 30 m DEM; bilinear resampling
    05_BoxPost       3x3 box car filter                       [--no-post-box]
    YYYYMMDD_MISSION_MODE/     e.g. 20250622_ALOS4_quad/
                     the final matrix bands as ENVI .bin/.hdr (byte order 0,
                     converted by py/snap2psp_inplace.py), no-data = NaN;
                     rgb/  RGB stack + PNG preview (quad: T22, T33, T11;
                           dual: C11, C22, |C12|);
                     processing.json  (options, SNAP version, looks, checks)
    alos_polsar_process.log    everything gpt printed (tail -f to watch)

    Names differ from alos2_dualpol_process.py's (01_Mlk ... 07_Box), so old
    outputs in the same folder are left alone. Do not run
    alos2_dualpol_delete_intermediary.py on these folders: it deletes
    everything that starts with 01 to 06. Use --delete-intermediates here.

Metadata checks (after the probe, and again on 01_CalMtxMlk)
    A dataset fails if any of these is wrong. SNAP 14.0.0-14.0.2 misreads
    ALOS-4 metadata and still reports success (GitHub
    senbox-org/microwave-toolbox issue #154, "ALOS-4 (PALSAR-3) CEOS reader
    misreads the leader"; fixed on master by PR #155 on Oct 30 2026, in no
    release yet). These checks catch that, and check the PALSAR-2 conversion:
      - MISSION in the metadata is the expected one (ALOS2 for ALOS-2 data and
        for converted ALOS-4 data, ALOS4 with --native-alos4)
      - range and azimuth pixel spacing greater than 0
      - slant range to the first pixel between 400 and 2500 km
      - near and far incidence angles between 5 and 75 degrees
      - PASS (ASCENDING / DESCENDING) agrees with the direction the
        satellite was moving (orbit state vectors), with the orbit direction
        letter in the original file name (ALOS-2: the last letter, as in
        ...HBQR1.1__A; ALOS-4: the letter after the look direction, as in
        ...HBQ-RD0606), and with the .kml or summary.txt if either of those
        spells the direction out
      - the polarisations SNAP read match the IMG- files
      - after 01_CalMtxMlk: every matrix band was written, at full size
    The step output that failed is deleted, so a later run (with a fixed
    SNAP) starts clean.

No-data and edges
    SNAP's box filter averages no-data (zero) into the pixels around it,
    which biases them low: at scene edges, and around areas
    Terrain-Flattening masks as layover or foreshortening (small ones can
    disappear altogether). So after each box filter, every pixel whose
    filter window held no-data in the filter's input is set back to
    no-data, in that step's product. Terrain-Correction's bilinear
    resampling also mixes no-data into output pixels within one input pixel
    of it: in the export those (1 or 2 output pixels from no-data, from the
    multilooked pixel size) are set to NaN. --no-edge-trim keeps them.

--tf-beta0-correction (off by default, until checked on one scene)
    SNAP's ALOS calibration scales complex data to sigma0, while
    Terrain-Flattening treats its input as beta0. The flattened matrix
    therefore comes out multiplied by sin(ellipsoid incidence angle): about
    -1.7 dB for ALOS-4 SM2, -2 to -3 dB for ALOS-2. This option divides the
    exported matrix by that factor, using the incidence angle that
    Terrain-Correction saves. It changes Terrain-Correction, so switching it
    on after a run needs --redo.

Stopping and resuming
    Safe to stop at any time (Ctrl-C, kill, closed terminal) and run again:
    each dataset carries on from its newest complete step. (After kill -9,
    gpt may keep running on its own: stop it before starting again.)
    -j (number of datasets at a time) and --delete-intermediates can differ
    between runs. Each step writes into a hidden ".partial_<step>" folder
    inside the dataset folder and is moved into place only after gpt
    succeeds and the output checks out, so any output that exists is
    complete. Leftover ".partial_*" folders are deleted on the next run.
    Do not run two copies on the same folder at the same time.
    Only what is missing is made again, from the nearest product still
    there: a missing PNG / RGB preview from the exported bands (no SNAP
    step runs); a missing export band from the final SNAP product; a missing
    final product from the newest step product left (or, after
    --delete-intermediates, from the original data). The same in ./warp/:
    a missing preview, PNG copy or GIF is made on its own, and a warped
    dataset is only resampled again if its bands are missing or its export
    holds different data (made from a newer final product).
    The PNG title is the original product ID; the caption gives the data
    attribution (JAXA, year), acquisition time (UTC) and pass direction,
    observation mode, incidence range, and the map grid (or, in ./warp/,
    the reference date). PNGs with other labels are made again.
    The options a dataset was processed with are saved in
    alos_polsar_settings.json. If a later run asks for different processing
    options, that dataset fails with a message instead of mixing outputs;
    --redo deletes this script's outputs for each dataset and starts again
    (if a --redo run is stopped, carry on without --redo; datasets it had
    not reached yet keep their earlier outputs).

--delete-intermediates
    Deletes each intermediate as soon as nothing still to be done needs it:
    the PALSAR-2 format copy of ALOS-4 data once 01_CalMtxMlk is made, and
    each step product once the next step's product is made (04_TC is kept
    for --tf-beta0-correction). The final product (05_BoxPost) is kept: the
    export is made from it. Turned on in a later run, it first deletes what
    earlier runs left. A step is only deleted after the step that reads it
    has finished, so stopping at any point loses at most the step running.
    --delete-intermediates-only does the same deleting without processing.

Warp (at the end of every run; --no-warp turns it off)
    Puts every finished output (from this run and earlier ones) on one
    common grid for a time series, in ./warp/ (this was alos4_warp.py):
        <name>/            the bands on the reference grid, ENVI .bin/.hdr,
                           no-data = NaN; rgb/ RGB stack + PNG preview;
                           warp.json (source, reference, resampling)
        <name>.png         copy of each PNG preview
        sequence_quad.gif  quad-pol previews in date order (T22, T33, T11)
        sequence_dual.gif  dual-pol previews in date order (C11, C22, |C12|)
        reference.txt      which dataset defines the grid
        alos4_warp.log     output of the tools that were run
    The reference is the earliest date, or --warp-ref (a date such as
    20250622, or a name such as 20250622_ALOS4_quad). Its bands are copied;
    every other dataset's are resampled onto its grid with
    py/raster_project_onto.py (GDAL). All outputs are snapped to the same
    20 m grid, so the default nearest-neighbour resampling only moves whole
    pixels and keeps values unchanged; datasets not on the reference's grid
    are reported (--warp-bilinear may suit those better). Quad-pol and
    dual-pol go on the same grid but are animated separately. Two datasets
    with the same name (same date, mission and mode) are an error.
    Resumable like the rest: finished ones are skipped, unless the dataset
    was processed again since or the resampling changed; if the reference
    changes (e.g. an earlier date was processed since), ./warp/ is made
    again from the start (--warp-ref keeps the old reference). It runs even
    when some datasets failed, on the ones that are finished.
    --warp-only does just this step.

Usage (run from the folder that holds the dataset folders)
    python3 alos4_process.py                    # all defaults
    python3 alos4_process.py --dry-run          # list datasets, plan
    python3 alos4_process.py -j 4               # 4 datasets at a time
    python3 alos4_process.py --no-tf --no-pre-box
    python3 alos4_process.py --delete-intermediates
    python3 alos4_process.py --delete-intermediates-only
    python3 alos4_process.py --redo
    python3 alos4_process.py --no-warp          # skip the warp at the end
    python3 alos4_process.py --warp-only --warp-ref 20250622
    python3 alos4_process.py -h                 # all options

Needs (as the older rcm scripts do): wps-research/py on the same checkout
(misc.py, snap2psp_inplace.py, raster_stack.py, raster_plot.py,
raster_project_onto.py) and the
wps-research C++ tools sbo_inplace.exe, convert_iq_to_cplx and abs on the
PATH (or compiled in wps-research/cpp). Exit code 0 when nothing failed.
"""

import io
import os
import sys
import json
import time
import math
import shutil
import struct
import contextlib
import signal
import argparse
import datetime
import threading
import traceback
import subprocess
import multiprocessing
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from xml.sax.saxutils import escape

sep = os.path.sep
my_path = os.path.dirname(os.path.abspath(__file__)) + sep
sys.path.append(my_path + '..' + sep + 'py')
import misc                                   # wps-research/py/misc.py
from misc import find_snap, err
import numpy as np

SCRIPT_VERSION = '20261102'

VOL_PREFIXES = ('VOL-ALOS2', 'VOL-ALOS4')
POLS = ('HH', 'HV', 'VH', 'VV')
BANDS = {'quad': ['T11', 'T12_real', 'T12_imag', 'T13_real', 'T13_imag',
                  'T22', 'T23_real', 'T23_imag', 'T33'],
         'dual': ['C11', 'C12_real', 'C12_imag', 'C22']}
DIAG = {'quad': ['T11', 'T22', 'T33'], 'dual': ['C11', 'C22']}
MATRIX = {'quad': 'T3', 'dual': 'C2'}
RGB_BANDS = {'quad': 'T22, T33, T11', 'dual': 'C11, C22, |C12|'}

DEM = 'Copernicus 30m Global DEM'
RESAMPLING = 'BILINEAR_INTERPOLATION'
BOX_FILTER = 'Box Car Filter'
INC_BAND = 'incidenceAngleFromEllipsoid'
PROBE_SIZE = 64

STEP_NAMES = ['01_CalMtxMlk', '02_TF', '03_BoxPre', '04_TC', '05_BoxPost']
TC_STEP = '04_TC'
LOG_NAME = 'alos_polsar_process.log'
SETTINGS_NAME = 'alos_polsar_settings.json'
SCENE_NAME = 'alos_polsar_scene.json'
GRAPH_NAME = '01_CalMtxMlk_graph.xml'
PARTIAL = '.partial_'

EXPORT_ONLY_KEYS = ['edge_trim']    # changing these only redoes the export

# plausible ranges used by the metadata checks
SLANT_RANGE_MIN, SLANT_RANGE_MAX = 400e3, 2500e3     # metres
INCIDENCE_MIN, INCIDENCE_MAX = 5.0, 75.0             # degrees

PRINT_LOCK = None


class Failure(Exception):
    pass


class Skip(Exception):
    pass


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def say(text):
    """Print one line; a lock keeps lines from parallel workers whole."""
    if PRINT_LOCK is not None:
        with PRINT_LOCK:
            print(text, flush=True)
    else:
        print(text, flush=True)


def fmt_time(seconds):
    s = int(round(seconds))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return '%dh%02dm%02ds' % (h, m, s) if h else '%dm%02ds' % (m, s)


def now_str():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_json(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')
    os.replace(tmp, path)


def write_text(path, text):
    with open(path, 'w') as f:
        f.write(text)


def remove_path(p):
    if os.path.isdir(p) and not os.path.islink(p):
        shutil.rmtree(p)
    elif os.path.lexists(p):
        os.remove(p)


def to_float(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def parse_utc(s):
    """SNAP time string '22-JUN-2025 12:34:56.123456' -> datetime (or None)."""
    if not s:
        return None
    s = s.strip()
    whole, frac = s, ''
    if '.' in s:
        whole, frac = s.split('.', 1)
    try:
        t = datetime.datetime.strptime(whole, '%d-%b-%Y %H:%M:%S')
    except ValueError:
        return None
    if frac.isdigit():
        t += datetime.timedelta(seconds=float('0.' + frac))
    return t


def is_export_name(name):
    """True for names like 20250622_ALOS4_quad (no regular expressions)."""
    parts = name.split('_')
    return (len(parts) == 3 and len(parts[0]) == 8 and parts[0].isdigit() and
            parts[1] in ('ALOS2', 'ALOS4') and parts[2] in ('quad', 'dual'))


def export_dirs(path):
    """Complete export folders inside a dataset folder."""
    try:
        names = sorted(os.listdir(path))
    except OSError:
        return []
    return [os.path.join(path, n) for n in names
            if is_export_name(n) and os.path.isdir(os.path.join(path, n))]


def read_envi_header(hdr):
    """The few ENVI header fields used here, as a dict of strings."""
    out = {}
    with open(hdr) as f:
        for line in f:
            if '=' in line:
                k, v = line.split('=', 1)
                k = k.strip().lower()
                if k in ('samples', 'lines', 'bands', 'data type', 'byte order',
                         'header offset', 'interleave'):
                    out[k] = v.strip()
    return out


# ---------------------------------------------------------------------------
# Finding and classifying datasets
# ---------------------------------------------------------------------------
def vol_files_in(d):
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return []
    return [n for n in names
            if n.startswith(VOL_PREFIXES) and os.path.isfile(os.path.join(d, n))]


def find_datasets(root='.'):
    """Folders in root with a VOL-ALOS2*/VOL-ALOS4* file directly inside
    (or, EODMS layout, inside X/X/). Returns (datasets, ignored_names)."""
    datasets, ignored = [], []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if name.startswith('.') or not os.path.isdir(p):
            continue
        path, vols = p, vol_files_in(p)
        if not vols:
            inner = os.path.join(p, name)
            if os.path.isdir(inner):
                vols = vol_files_in(inner)
                if vols:
                    path = inner
        if not vols:
            ignored.append(name)
            continue
        datasets.append({'name': name, 'path': os.path.abspath(path),
                         'vols': vols, 'vol': vols[0]})
    return datasets, ignored


def classify(ds, convert_alos4=True):
    """Fill in mission, polarisations and mode from the file names alone.
    Returns None if the dataset can be processed, else a reason to skip."""
    if len(ds['vols']) > 1:
        return 'more than one VOL file: %s' % ', '.join(ds['vols'])
    ds['mission'] = 'ALOS2' if ds['vol'].startswith('VOL-ALOS2') else 'ALOS4'
    counts = {}
    for n in sorted(os.listdir(ds['path'])):
        if n.startswith('IMG-') and os.path.isfile(os.path.join(ds['path'], n)):
            parts = n.split('-')
            pol = parts[1].upper() if len(parts) > 2 else '?'
            counts[pol] = counts.get(pol, 0) + 1
    if not counts:
        return 'no IMG- (image) files'
    unknown = sorted(p for p in counts if p not in POLS)
    if unknown:
        return 'image files with unexpected polarisation(s): %s' % ', '.join(unknown)
    if any(c > 1 for c in counts.values()):
        return ('more than one image file per polarisation '
                '(multi-beam ScanSAR Level 1.1): only stripmap is processed')
    ds['pols'] = [p for p in POLS if p in counts]
    n = len(ds['pols'])
    if n == 4:
        ds['mode'] = 'quad'
    elif n == 2:
        ds['mode'] = 'dual'
    elif n == 1:
        return 'single-pol (%s): only dual-pol and quad-pol are processed' % ds['pols'][0]
    else:
        return 'unexpected polarisations: %s' % ', '.join(ds['pols'])
    ds['convert'] = ds['mission'] == 'ALOS4' and convert_alos4
    if ds['convert']:
        return p3_plan(ds)
    return None


# ---------------------------------------------------------------------------
# PALSAR-3 -> PALSAR-2 CEOS conversion
#
# Port of JAXA's sample code P3toP2snap (Sep 2025, "converting PALSAR-3 L1.1
# standard products to PALSAR-2 CEOS format", stripmap 3 m / 6 m / 10 m),
# author Dr. Yunung Nina Lin:
#     https://www.eorc.jaxa.jp/ALOS/jp/alos-4/tool/P3toP2snap.zip
# Thanks to JAXA for making this code publicly available.
# Made so SNAP can read ALOS-4 data with its ALOS-2 reader. Same byte edits
# and file names as JAXA's code. Differences: image files are converted line
# by line as a stream (JAXA's reads each whole file into memory and writes it
# out three times); every record length is checked against the fixed layout
# the code assumes, and the conversion stops if one differs; the second image
# file pointer of dual-pol data is renamed like the others (JAXA's test for
# it compares the ALOS-4 mode with the ALOS-2 name "FBD", so it never runs).
# The ALOS-2 style names always end in "A" (as in JAXA's code), whatever the
# orbit direction: the real direction stays in the leader and text record.
# ---------------------------------------------------------------------------
P2_DIR = 'ALOS2_format'                   # JAXA's name for the converted copy
P3_MODES = {'UWS': 'UBS', 'FWD': 'FBD', 'HBQ': 'HBQ', 'UBD': 'UBD', 'UWD': 'UBD',
            'UBQ': 'UBQ'}                 # ALOS-4 mode -> ALOS-2 mode (JAXA's table)
P3_LED_RECORDS = [720, 4680, 4680, 16384, 65536, 1620, 325000, 511000, 3072, 728000, 9500]
P3_LINE_PREFIX, P2_LINE_PREFIX = 800, 544         # bytes before the pixels of a line
CEOS_VOL_RECORD = 360


def p2_name(name, mode2):
    """JAXA's ALOS-2 style file name for a PALSAR-3 file name."""
    k = 7 if name.startswith('IMG-') else 4    # 'IMG-HH-' or 'VOL-' / 'LED-' / 'TRL-'
    return (name[:k + 4] + '240' + name[k + 5:k + 12] + '-' + name[k + 12:k + 18] + '-' +
            mode2 + 'R1.1__A')


def p3_plan(ds):
    """Check that the dataset fits JAXA's conversion and work out the names.
    Returns None if it does, else a reason to skip."""
    vol = ds['vol']                 # e.g. VOL-ALOS41712610250622HBQ-RD0606-1.1__-
    mode = vol[22:25]
    if not (len(vol) >= 28 and vol[4:9] == 'ALOS4' and vol[9:22].isdigit() and
            vol[27:28] in ('A', 'D')):
        return ('VOL file name %s is not in the form JAXA\'s PALSAR-3 to PALSAR-2 '
                'conversion expects' % vol)
    if mode not in P3_MODES:
        return ('observation mode %s: JAXA\'s PALSAR-3 to PALSAR-2 conversion only covers '
                'the stripmap modes %s' % (mode, ', '.join(sorted(P3_MODES))))
    if {'S': 1, 'D': 2, 'Q': 4}.get(mode[2]) != len(ds['pols']):
        return 'mode %s does not match the %d image files' % (mode, len(ds['pols']))
    suffix = vol[4:]
    files = {'LED': 'LED-' + suffix, 'TRL': 'TRL-' + suffix}
    files.update({p: 'IMG-%s-%s' % (p, suffix) for p in ds['pols']})
    missing = [f for f in files.values() if not os.path.isfile(os.path.join(ds['path'], f))]
    if missing:
        return 'files needed for the PALSAR-2 conversion are missing: %s' % ', '.join(missing)
    mode2 = P3_MODES[mode]
    ds['p3'] = {'mode': mode, 'mode2': mode2, 'look': vol[26:27], 'node': vol[27:28],
                'scene_id': vol[4:8] + '240' + vol[9:16] + '-' + vol[16:22],
                'files': files,
                'vol2': p2_name(vol, mode2)}
    return None


def _put(buf, offset, data):
    """Overwrite bytes in place (like JAXA's edit_byte), never past the end."""
    if offset + len(data) > len(buf):
        raise Failure('PALSAR-2 conversion: edit past the end of a record')
    buf[offset:offset + len(data)] = data


def _records(path, sizes, what):
    """Split a CEOS file into records of the given sizes, checking each
    record's own length field (bytes 8-11)."""
    with open(path, 'rb') as f:
        data = f.read(sum(sizes))
    if len(data) < sum(sizes):
        raise Failure('%s is %d bytes, JAXA\'s layout needs %d' % (what, len(data), sum(sizes)))
    out, pos = [], 0
    for i, n in enumerate(sizes):
        r = bytearray(data[pos:pos + n])
        length = struct.unpack('>L', r[8:12])[0]
        if length != n:
            raise Failure('%s record %d is %d bytes long, JAXA\'s layout says %d' % (
                what, i + 1, length, n))
        out.append(r)
        pos += n
    return out


def p3_convert_vol(src, dst, p3, n_img):
    recs = _records(src, [CEOS_VOL_RECORD] * (n_img + 4), 'VOL file')
    desc, pole, poims, potr, trec = recs[0], recs[1], recs[2:2 + n_img], recs[-2], recs[-1]
    _put(desc, 60, b'AL2')
    _put(desc, 76, b'ALOS2')
    for r in [pole] + poims + [potr]:
        _put(r, 20, b'AL2')
    pid = 'PRODUCT:' + p3['mode2'] + p3['look'] + '1.1__' + p3['node']
    _put(trec, 16, pid.encode() + b' ' * 22)
    _put(trec, 163, p3['scene_id'].encode() + b' ' * 12)
    with open(dst, 'wb') as f:
        for r in recs:
            f.write(r)


def p3_convert_led(src, dst, p3):
    (desc, summ, posi, attd, radi, qual,
     fac1, fac2, fac3, fac4, fac5) = _records(src, P3_LED_RECORDS, 'LED file')
    # file descriptor record: the PALSAR-2 record sizes and counts
    _put(desc, 48, b'AL2 SARBSARL    ')
    _put(desc, 186, b'  4096')
    _put(desc, 234, b'  9860')
    for o in (342, 348, 354):
        _put(desc, o, b'     0')
    _put(desc, 360, b' ' * 60)
    for o, v in ((420, b'     1'), (426, b'  325000'), (434, b'     1'), (440, b'  511000'),
                 (448, b'     1'), (454, b'    3072'), (462, b'     1'), (468, b'  728000'),
                 (476, b'     1'), (482, b'    5000')):
        _put(desc, o, v)
    _put(desc, 490, b' ' * 230)
    # data set summary record: 4680 bytes (PALSAR-3) -> 4096 (PALSAR-2)
    mode = p3['mode']
    if mode[0] == 'U':
        ope_mode = b'01'
    elif mode[0] == 'F':
        ope_mode = b'03'
    elif mode == 'HBQ':
        ope_mode = b'18'
    else:
        raise Failure('PALSAR-2 conversion: no operation mode code for %s' % mode)
    orig = bytes(summ)
    _put(summ, 8, struct.pack('>L', 4096))
    _put(summ, 388, b'   ' + (b'4' if mode[2] == 'S' else b'8'))
    _put(summ, 396, b'ALOS2' + b' ' * 11)
    _put(summ, 20, p3['scene_id'].encode() + b' ' * 11)
    _put(summ, 416, b'2')
    _put(summ, 422, ope_mode)
    # fields 105-167: from their PALSAR-3 offsets to the PALSAR-2 ones (JAXA's table)
    for a, b, to in ((1622, 1670, 1478), (1822, 1830, 1534), (1830, 1878, 1542),
                     (2038, 2086, 1606), (2246, 2254, 1670), (2254, 2258, 1678),
                     (2274, 2278, 1682), (2282, 2298, 1686), (2298, 2314, 1702),
                     (2314, 2330, 1718), (2330, 2346, 1734), (2346, 2362, 1750),
                     (2362, 2366, 1766), (2366, 2374, 1770), (2374, 2382, 1778),
                     (2382, 2390, 1786), (2390, 2398, 1794), (2398, 2402, 1802),
                     (2402, 2410, 1806), (2410, 2426, 1814), (2426, 2430, 1830),
                     (2434, 2450, 1838), (2450, 2454, 1854), (2482, 2502, 1886),
                     (2502, 2522, 1906), (2522, 2542, 1926), (2542, 2562, 1946),
                     (2562, 2582, 1966), (2582, 2602, 1986), (2602, 4680, 2006)):
        _put(summ, to, orig[a:b])
    for o, n in ((1462, 16), (1526, 8), (1590, 16), (1654, 16), (1834, 4), (1858, 28),
                 (4084, 12)):
        _put(summ, o, b' ' * n)
    _put(radi, 8, struct.pack('>L', 9860))
    _put(fac5, 8, struct.pack('>L', 5000))
    with open(dst, 'wb') as f:
        for r in (desc, summ[:4096], posi, attd, radi[:9860], qual, fac1, fac2, fac3, fac4,
                  fac5[:5000]):
            f.write(r)


def p3_convert_trl(src, dst):
    with open(src, 'rb') as f:
        d = bytearray(f.read(720))
    if len(d) != 720:
        raise Failure('TRL file is shorter than 720 bytes')
    _put(d, 48, b'AL2')
    _put(d, 490, b'     1')
    _put(d, 516, b'     2')
    with open(dst, 'wb') as f:
        f.write(d)


def p3_img_layout(src):
    """(record length, pixels per line, lines) of a PALSAR-3 image file."""
    size = os.path.getsize(src)
    with open(src, 'rb') as f:
        f.seek(720)
        first = f.read(P3_LINE_PREFIX)
    if len(first) < 28:
        raise Failure('%s has no image lines' % os.path.basename(src))
    rec_len = struct.unpack('>L', first[8:12])[0]
    rec_pix = struct.unpack('>L', first[24:28])[0]
    if rec_len != P3_LINE_PREFIX + 8 * rec_pix or (size - 720) % rec_len:
        raise Failure('%s: lines of %d bytes for %d pixels do not fit JAXA\'s layout' % (
            os.path.basename(src), rec_len, rec_pix))
    return rec_len, rec_pix, (size - 720) // rec_len


def p3_convert_img(src, dst):
    """Line by line: an 800-byte line header becomes the first 544 bytes of
    it, with the record length cut by 256, bytes 32-35 zeroed, and the slant
    range to the first pixel (bytes 116-119) from centimetres to metres."""
    rec_len, rec_pix, lines = p3_img_layout(src)
    npix = 8 * rec_pix
    with open(src, 'rb', buffering=16 << 20) as fi, open(dst, 'wb', buffering=16 << 20) as fo:
        desc = bytearray(fi.read(720))
        try:
            fn30 = int(desc[186:192]) - 256     # field 30: record length
        except ValueError:
            raise Failure('%s: unreadable record length in the file descriptor' %
                          os.path.basename(src))
        _put(desc, 48, b'AL2 SARBIMOP    ')
        _put(desc, 186, b'%6d' % fn30)
        _put(desc, 276, b' 544')
        fo.write(desc)
        for i in range(lines):
            rec = fi.read(rec_len)
            if len(rec) != rec_len or struct.unpack('>L', rec[8:12])[0] != rec_len:
                raise Failure('%s: line %d is not %d bytes long' % (
                    os.path.basename(src), i + 1, rec_len))
            pre = bytearray(rec[:P2_LINE_PREFIX])
            _put(pre, 8, struct.pack('>L', rec_len - 256))
            _put(pre, 32, struct.pack('>L', 0))
            slant_cm = struct.unpack('>L', rec[116:120])[0]
            _put(pre, 116, struct.pack('>L', slant_cm // 100))
            fo.write(pre)
            fo.write(rec[P3_LINE_PREFIX:P3_LINE_PREFIX + npix])
    return lines


def p2_expected(ds):
    """Converted file name -> expected size in bytes."""
    p3, path = ds['p3'], ds['path']
    out = {p3['vol2']: CEOS_VOL_RECORD * (len(ds['pols']) + 4),
           p2_name(p3['files']['LED'], p3['mode2']):
               sum(P3_LED_RECORDS) - (4680 - 4096) - (65536 - 9860) - (9500 - 5000),
           p2_name(p3['files']['TRL'], p3['mode2']): 720}
    for p in ds['pols']:
        rec_len, rec_pix, lines = p3_img_layout(os.path.join(path, p3['files'][p]))
        out[p2_name(p3['files'][p], p3['mode2'])] = 720 + lines * (rec_len - 256)
    return out


def p2_complete(ds, folder):
    try:
        expected = p2_expected(ds)
    except (Failure, OSError):
        return False
    return all(os.path.isfile(os.path.join(folder, n)) and
               os.path.getsize(os.path.join(folder, n)) == size
               for n, size in expected.items())


def p3_to_p2(ds, log):
    """The dataset in PALSAR-2 format in <dataset>/ALOS2_format/ (made if
    missing or incomplete). Returns the path of its VOL file."""
    path, p3 = ds['path'], ds['p3']
    out = os.path.join(path, P2_DIR)
    if os.path.isdir(out) and p2_complete(ds, out):
        return os.path.join(out, p3['vol2'])
    staging = new_staging(path, P2_DIR)
    try:
        src = lambda name: os.path.join(path, name)
        dst = lambda name: os.path.join(staging, p2_name(name, p3['mode2']))
        p3_convert_vol(src(ds['vol']), dst(ds['vol']), p3, len(ds['pols']))
        p3_convert_led(src(p3['files']['LED']), dst(p3['files']['LED']), p3)
        p3_convert_trl(src(p3['files']['TRL']), dst(p3['files']['TRL']))
        for p in ds['pols']:
            lines = p3_convert_img(src(p3['files'][p]), dst(p3['files'][p]))
            log.write('converted %s (%d lines)\n' % (p3['files'][p], lines))
        if not p2_complete(ds, staging):
            raise Failure('PALSAR-2 conversion: converted files have unexpected sizes')
        remove_path(out)
        os.rename(staging, out)
    except BaseException:
        remove_path(staging)
        raise
    return os.path.join(out, p3['vol2'])


# ---------------------------------------------------------------------------
# Processing plan and settings
# ---------------------------------------------------------------------------
def settings_from_args(a):
    st = {'range_looks': a.rg_looks,
          'azimuth_looks': a.az_looks if a.az_looks else 'derived (square ground pixels)',
          'terrain_flattening': not a.no_tf,
          'tf_beta0_correction': bool(a.tf_beta0_correction),
          'pre_tc_box': 0 if a.no_pre_box else a.pre_box_size,
          'post_tc_box': 0 if a.no_post_box else a.post_box_size,
          'map_projection': a.crs,
          'pixel_spacing_m': a.pixel_spacing,
          'grid_aligned_to_m': [0.0, 0.0],
          'dem': DEM,
          'resampling': RESAMPLING,
          'edge_trim': not a.no_edge_trim,
          'alos4_input': ('SNAP ALOS-4 reader' if a.native_alos4 else
                          'converted to PALSAR-2 format (JAXA P3toP2snap code)')}
    return json.loads(json.dumps(st))       # normalise (as stored in JSON)


def plan_steps(mode, st):
    """The SNAP steps for one dataset, in order, for the given settings."""
    bands = BANDS[mode]
    fixed = st['azimuth_looks'] if isinstance(st['azimuth_looks'], int) else None
    steps = [{'name': '01_CalMtxMlk', 'kind': 'graph',
              'what': 'Calibration + %s matrix + Multilook' % MATRIX[mode],
              'bands': bands,
              # for the record only: the graph itself is written by graph_cal_mtx_mlk()
              'params': [('Calibration.outputImageInComplex', 'true'),
                         ('Polarimetric-Matrices.matrix', MATRIX[mode]),
                         ('Multilook.nRgLooks', str(st['range_looks'])),
                         ('Multilook.nAzLooks', str(fixed) if fixed else 'derived'),
                         ('Multilook.outputIntensity', 'false'),
                         ('Multilook.grSquarePixel', 'false' if fixed else 'true')]}]
    if st['terrain_flattening']:
        steps.append({'name': '02_TF', 'kind': 'op', 'op': 'Terrain-Flattening',
                      'what': 'Terrain-Flattening', 'bands': bands,
                      'params': [('demName', st['dem']),
                                 ('demResamplingMethod', st['resampling']),
                                 ('nodataValueAtSea', 'true')]})
    if st['pre_tc_box']:
        n = st['pre_tc_box']
        steps.append({'name': '03_BoxPre', 'kind': 'op', 'box': n,
                      'op': 'Polarimetric-Speckle-Filter',
                      'what': '%dx%d box filter (before TC)' % (n, n), 'bands': bands,
                      'params': [('filter', BOX_FILTER), ('filterSize', str(n))]})
    tc_params = [('demName', st['dem']),
                 ('demResamplingMethod', st['resampling']),
                 ('imgResamplingMethod', st['resampling']),
                 ('mapProjection', st['map_projection']),
                 ('pixelSpacingInMeter', repr(float(st['pixel_spacing_m']))),
                 ('alignToStandardGrid', 'true'),
                 ('standardGridOriginX', repr(float(st['grid_aligned_to_m'][0]))),
                 ('standardGridOriginY', repr(float(st['grid_aligned_to_m'][1]))),
                 ('nodataValueAtSea', 'true')]
    tc_bands = list(bands)
    if st['tf_beta0_correction']:
        tc_params.append(('saveIncidenceAngleFromEllipsoid', 'true'))
        tc_bands.append(INC_BAND)
    steps.append({'name': TC_STEP, 'kind': 'op', 'op': 'Terrain-Correction',
                  'what': 'Terrain-Correction (%s, %g m)' % (st['map_projection'],
                                                             st['pixel_spacing_m']),
                  'bands': tc_bands, 'params': tc_params})
    if st['post_tc_box']:
        n = st['post_tc_box']
        steps.append({'name': '05_BoxPost', 'kind': 'op', 'box': n,
                      'op': 'Polarimetric-Speckle-Filter',
                      'what': '%dx%d box filter (after TC)' % (n, n), 'bands': bands,
                      'params': [('filter', BOX_FILTER), ('filterSize', str(n))]})
    return steps


def dim_of(path, name):
    return os.path.join(path, name + '.dim')


def step_exists(path, name):
    return os.path.isfile(dim_of(path, name))


def steps_to_run(path, steps, st):
    """Indices of the steps that must run so the final product (and, for the
    beta0 correction, the Terrain-Correction product) exist."""
    have = [step_exists(path, s['name']) for s in steps]
    names = [s['name'] for s in steps]
    required = [len(steps) - 1]
    if st['tf_beta0_correction']:
        required.append(names.index(TC_STEP))
    run = set()
    for i in required:
        while i >= 0 and not have[i]:
            run.add(i)
            i -= 1
    return sorted(run)


def kept_steps(steps, st):
    keep = {steps[-1]['name']}
    if st.get('tf_beta0_correction'):
        keep.add(TC_STEP)
    return keep


def outputs_present(path):
    return (any(step_exists(path, n) for n in STEP_NAMES) or
            len(export_dirs(path)) > 0)


def check_settings(path, st, log, mission):
    """Refuse to mix outputs made with different options."""
    f = os.path.join(path, SETTINGS_NAME)
    old = read_json(f)
    if old is None:
        if outputs_present(path):
            raise Failure('outputs exist but %s is missing, so the options they '
                          'were made with are unknown; use --redo' % SETTINGS_NAME)
    else:
        ignore = set(EXPORT_ONLY_KEYS)
        if mission != 'ALOS4':
            ignore.add('alos4_input')
        diff = [k for k in st if k not in ignore and old.get(k) != st[k]]
        if diff and outputs_present(path):
            raise Failure('existing outputs were made with other options (%s). Rerun '
                          'with those options, or use --redo to delete this script\'s '
                          'outputs for the dataset and start again' %
                          '; '.join('%s: %s -> %s' % (k, old.get(k), st[k]) for k in diff))
        diff_e = [k for k in EXPORT_ONLY_KEYS if old.get(k) != st.get(k)]
        if diff_e:
            for d in export_dirs(path):
                log.write('export option changed (%s): removing %s\n' %
                          (', '.join(diff_e), os.path.basename(d)))
                remove_path(d)
    write_json(f, st)


# ---------------------------------------------------------------------------
# Running gpt and other tools
# ---------------------------------------------------------------------------
def kill_group(p):
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except OSError:
        return
    try:
        p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
        p.wait()


def run_logged(cmd, log, cwd=None, env=None):
    """Run a command in its own process group with its output going to the
    log. If this process is interrupted, the command is stopped too."""
    log.write('\n[%s] $ %s\n' % (now_str(), ' '.join(quote(c) for c in cmd)))
    if cwd:
        log.write('    (in %s)\n' % cwd)
    log.flush()
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        rc = p.wait()
    except BaseException:
        kill_group(p)
        raise
    log.write('[%s] exit code %d\n' % (now_str(), rc))
    log.flush()
    return rc


def quote(s):
    s = str(s)
    if s and all(c.isalnum() or c in '-_./=:,+@%' for c in s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def gpt_extra(ctx):
    extra = ['-e']                              # full error messages in the log
    if ctx.get('threads'):
        extra += ['-q', str(ctx['threads'])]
    return extra


def gpt_version(gpt, timeout=300):
    """(major, minor, patch) from `gpt --diag`, or None."""
    marker = 'SNAP Release version'
    try:
        p = subprocess.Popen([gpt, '--diag'], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             text=True, errors='replace', start_new_session=True)
    except OSError:
        return None
    found = {}

    def reader():
        for line in p.stdout:
            i = line.find(marker)
            if i >= 0 and 'v' not in found:
                found['v'] = line[i + len(marker):].strip()
                found['t'] = time.time()

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    t0 = time.time()
    while p.poll() is None:
        if 'v' in found and time.time() - found['t'] > 5:
            break
        if time.time() - t0 > timeout:
            break
        time.sleep(0.2)
    if p.poll() is None:
        kill_group(p)
    t.join(timeout=5)
    s = found.get('v', '').split()
    if not s:
        return None
    parts = s[0].split('.')
    if not all(x.isdigit() for x in parts):
        return None
    return tuple(int(x) for x in parts)


def vstr(v):
    return '.'.join(str(x) for x in v) if v else 'unknown'


def pick_gpt(user_gpt):
    """gpt from --gpt; else the first SNAP 14+ among misc.find_snap(), every
    "gpt" on the PATH (snap_install.py adds SNAP's bin folder at the end of
    the PATH, so another program called gpt may come first) and ~/esa-snap.
    SNAP 14 or newer is required (ALOS-4 support)."""
    cands = []                                  # (path, is a known SNAP location)
    if user_gpt:
        cands.append((os.path.abspath(os.path.expanduser(user_gpt)), True))
    else:
        try:
            with contextlib.redirect_stdout(io.StringIO()):     # its "not found" message
                cands.append((find_snap(), True))
        except SystemExit:                      # find_snap() exits if nothing found
            pass
        for d in os.environ.get('PATH', '').split(os.pathsep):
            if d:
                cands.append((os.path.join(d, 'gpt'), False))
        cands.append((os.path.join(os.path.expanduser('~'), 'esa-snap', 'bin', 'gpt'), True))
    seen, found = set(), []
    for c, known in cands:
        c = os.path.realpath(c)
        if c in seen or not (os.path.isfile(c) and os.access(c, os.X_OK)):
            continue
        seen.add(c)
        print('checking SNAP version of', c, '(gpt --diag) ...', flush=True)
        found.append((c, gpt_version(c), known))
    for c, v, known in found:
        if v is not None and v[0] >= 14:
            return c, v
    for c, v, known in found:
        # a "gpt" from the PATH whose version cannot be read may not be SNAP's
        if v is None and known:
            print('WARNING: could not read the SNAP version of %s; using it anyway' % c)
            return c, v
    if found:
        err('SNAP 14 or newer is needed (ALOS-4 support); found: ' +
            ', '.join('%s (SNAP %s)' % (c, vstr(v)) for c, v, k in found) +
            '. Install it with snap_install.py, or point to it with --gpt.')
    err('SNAP gpt not found. Install it with snap_install.py, or use --gpt PATH.')


def find_tools():
    """The external tools the export uses. Returns (tools, env, missing)."""
    env = dict(os.environ)
    extra = [os.path.abspath(misc.cd), os.path.abspath(os.path.join(misc.pd, '..', 'bin'))]
    env['PATH'] = os.pathsep.join(extra + [env.get('PATH', '')])
    env['MPLBACKEND'] = 'Agg'                   # raster_plot.py without a display
    tools, missing = {}, []
    # snap2psp_inplace.py calls "sbo_inplace.exe" by that name
    if not shutil.which('sbo_inplace.exe', path=env['PATH']):
        missing.append('sbo_inplace.exe (used by snap2psp_inplace.py)')
    for name in ('convert_iq_to_cplx', 'abs'):
        p = (shutil.which(name, path=env['PATH']) or
             shutil.which(name + '.exe', path=env['PATH']))
        if p:
            tools[name] = p
        else:
            missing.append(name)
    for name in ('snap2psp_inplace.py', 'raster_stack.py', 'raster_plot.py'):
        p = os.path.join(misc.pd, name)
        if os.path.isfile(p):
            tools[name] = p
        else:
            missing.append(p)
    return tools, env, missing


# ---------------------------------------------------------------------------
# gpt graphs
# ---------------------------------------------------------------------------
def graph_xml(nodes):
    """nodes: list of (node id, operator, source node id or None, [(k, v)])."""
    out = ['<graph id="alos4_process">', '  <version>1.0</version>']
    for nid, op, src, params in nodes:
        out.append('  <node id="%s">' % nid)
        out.append('    <operator>%s</operator>' % op)
        if src:
            out.append('    <sources>')
            out.append('      <sourceProduct refid="%s"/>' % src)
            out.append('    </sources>')
        out.append('    <parameters>')
        for k, v in params:
            out.append('      <%s>%s</%s>' % (k, escape(str(v)), k))
        out.append('    </parameters>')
        out.append('  </node>')
    out.append('</graph>')
    return '\n'.join(out) + '\n'


def graph_probe(vol_path, target):
    return graph_xml([
        ('Read', 'Read', None, [('file', vol_path)]),
        ('Subset', 'Subset', 'Read', [('region', '0,0,%d,%d' % (PROBE_SIZE, PROBE_SIZE)),
                                       ('copyMetadata', 'true')]),
        ('Write', 'Write', 'Subset', [('file', target), ('formatName', 'BEAM-DIMAP')])])


def graph_cal_mtx_mlk(vol_path, target, mode, st):
    fixed = st['azimuth_looks'] if isinstance(st['azimuth_looks'], int) else None
    return graph_xml([
        ('Read', 'Read', None, [('file', vol_path)]),
        ('Calibration', 'Calibration', 'Read', [('outputImageInComplex', 'true')]),
        ('Matrix', 'Polarimetric-Matrices', 'Calibration', [('matrix', MATRIX[mode])]),
        ('Multilook', 'Multilook', 'Matrix',
         [('nRgLooks', str(st['range_looks'])),
          ('nAzLooks', str(fixed if fixed else 1)),
          ('outputIntensity', 'false'),     # matrix elements are averaged as they are
          ('grSquarePixel', 'false' if fixed else 'true')]),
        ('Write', 'Write', 'Multilook', [('file', target), ('formatName', 'BEAM-DIMAP')])])


# ---------------------------------------------------------------------------
# Reading SNAP (BEAM-DIMAP) products
# ---------------------------------------------------------------------------
DTYPE_BYTES = {'int8': 1, 'uint8': 1, 'int16': 2, 'uint16': 2, 'int32': 4,
               'uint32': 4, 'float32': 4, 'int64': 8, 'uint64': 8, 'float64': 8}


def read_dim(dim):
    """Size, bands and abstracted metadata of a BEAM-DIMAP product."""
    root = ET.parse(dim).getroot()

    def text(elem, tag, default=None):
        e = elem.find(tag)
        return e.text.strip() if e is not None and e.text else default

    info = {'ncols': int(text(root, 'Raster_Dimensions/NCOLS', '0')),
            'nrows': int(text(root, 'Raster_Dimensions/NROWS', '0')),
            'bands': {}, 'abs': {}, 'orbit': []}
    files = {}
    for df in root.iter('Data_File'):
        p, i = df.find('DATA_FILE_PATH'), df.find('BAND_INDEX')
        if p is not None and i is not None and i.text:
            files[i.text.strip()] = p.get('href')
    for sbi in root.iter('Spectral_Band_Info'):
        idx = text(sbi, 'BAND_INDEX')
        name = text(sbi, 'BAND_NAME')
        if name is None:
            continue
        info['bands'][name] = {
            'dtype': text(sbi, 'DATA_TYPE', 'float32'),
            'width': int(text(sbi, 'BAND_RASTER_WIDTH', str(info['ncols']))),
            'height': int(text(sbi, 'BAND_RASTER_HEIGHT', str(info['nrows']))),
            'virtual': text(sbi, 'VIRTUAL_BAND', 'false').lower() == 'true',
            'file': files.get(idx)}
    for e in root.iter('MDElem'):
        if e.get('name') == 'Abstracted_Metadata':
            for a in e.findall('MDATTR'):
                info['abs'][a.get('name')] = (a.text or '').strip()
            for o in e.findall('MDElem'):
                if o.get('name') == 'Orbit_State_Vectors':
                    for v in o.findall('MDElem'):
                        info['orbit'].append({a.get('name'): (a.text or '').strip()
                                              for a in v.findall('MDATTR')})
            break
    return info


def verify_product(dim, bands):
    """None if every listed band has its full-size data file, else a message."""
    if not os.path.isfile(dim):
        return 'gpt reported success but wrote no %s' % os.path.basename(dim)
    try:
        info = read_dim(dim)
    except (ET.ParseError, OSError, ValueError) as e:
        return 'cannot read %s (%s)' % (os.path.basename(dim), e)
    base = os.path.dirname(dim)
    for b in bands:
        bi = info['bands'].get(b)
        if bi is None:
            return 'band %s missing from %s' % (b, os.path.basename(dim))
        if bi['virtual'] or not bi['file']:
            return 'band %s has no data file' % b
        img = os.path.join(base, bi['file'])
        if img.endswith('.hdr'):
            img = img[:-4] + '.img'
        expected = bi['width'] * bi['height'] * DTYPE_BYTES.get(bi['dtype'], 4)
        if not os.path.isfile(img):
            return 'data file for band %s not written' % b
        if expected == 0 or os.path.getsize(img) != expected:
            return 'data file for band %s has %d bytes, expected %d' % (
                b, os.path.getsize(img), expected)
    return None


# ---------------------------------------------------------------------------
# Metadata checks
# ---------------------------------------------------------------------------
def pass_from_orbit(info):
    """ASCENDING / DESCENDING from the z velocity (Earth-fixed) of the orbit
    state vector nearest the first line, or None."""
    vecs = []
    for v in info['orbit']:
        z = to_float(v.get('z_vel'))
        if z is not None:
            vecs.append((parse_utc(v.get('time')), z))
    if not vecs:
        return None
    t0 = parse_utc(info['abs'].get('first_line_time'))
    timed = [(abs((t - t0).total_seconds()), z) for t, z in vecs if t and t0]
    if timed:
        z = min(timed)[1]
    else:
        signs = set(z > 0 for _, z in vecs if z != 0)
        if len(signs) != 1:
            return None
        z = vecs[0][1]
    if z == 0:
        return None
    return 'ASCENDING' if z > 0 else 'DESCENDING'


def pass_hints(ds):
    """Independent statements of the pass direction: [(source, PASS)]."""
    hints = []
    if ds['mission'] == 'ALOS2':
        # ALOS-2 product names end with the orbit direction, e.g. ...HBQR1.1__A
        if ds['vol'].endswith('__A'):
            hints.append((ds['vol'] + ' (last letter)', 'ASCENDING'))
        elif ds['vol'].endswith('__D'):
            hints.append((ds['vol'] + ' (last letter)', 'DESCENDING'))
    else:
        # ALOS-4 names: orbit direction after the look direction, e.g. HBQ-RD0606
        # (character 27 of the VOL name; JAXA's P3toP2snap code reads it there)
        node = ds['vol'][27:28]
        if node in ('A', 'D'):
            hints.append((ds['vol'] + ' (orbit direction letter)',
                          'ASCENDING' if node == 'A' else 'DESCENDING'))
    for n in sorted(os.listdir(ds['path'])):
        if not (n.lower().endswith('.kml') or n.lower() == 'summary.txt'):
            continue
        try:
            with open(os.path.join(ds['path'], n), errors='replace') as f:
                t = f.read().upper()
        except OSError:
            continue
        a, d = 'ASCENDING' in t, 'DESCENDING' in t
        if a != d:                             # exactly one of them is stated
            hints.append((n, 'ASCENDING' if a else 'DESCENDING'))
    return hints


def check_metadata(info, ds, stage):
    """Returns (problems, skip_reason)."""
    a = info['abs']
    problems, skip = [], None
    mission = a.get('MISSION', '')
    expected = 'ALOS2' if ds.get('convert') else ds['mission']   # converted: ALOS-2 reader
    if mission != expected:
        problems.append('MISSION is "%s", expected %s' % (mission, expected))
    acq = a.get('ACQUISITION_MODE', '')
    if acq in ('ScanSAR', 'Spotlight'):
        skip = 'acquisition mode %s: only stripmap is processed' % acq
    if stage == 'probe':
        if a.get('SAMPLE_TYPE', '').upper() != 'COMPLEX':
            skip = 'sample type %s: only complex (Level 1.1) data are processed' % (
                a.get('SAMPLE_TYPE', '?'))
        read_pols = sorted(n[2:] for n in info['bands'] if n.startswith('i_'))
        if read_pols != sorted(ds['pols']):
            problems.append('SNAP read polarisations %s, the IMG- files have %s' % (
                ','.join(read_pols) or 'none', ','.join(ds['pols'])))
    for key, label in (('range_spacing', 'range pixel spacing'),
                       ('azimuth_spacing', 'azimuth pixel spacing')):
        v = to_float(a.get(key))
        if v is None or not 0 < v < 1000:
            problems.append('%s is %s m (expected more than 0)' % (label, a.get(key)))
    sr = to_float(a.get('slant_range_to_first_pixel'))
    if sr is None or not SLANT_RANGE_MIN < sr < SLANT_RANGE_MAX:
        problems.append('slant range to first pixel is %s m (expected %d to %d km)' % (
            a.get('slant_range_to_first_pixel'), SLANT_RANGE_MIN / 1e3, SLANT_RANGE_MAX / 1e3))
    for key in ('incidence_near', 'incidence_far'):
        v = to_float(a.get(key))
        if v is None or not INCIDENCE_MIN < v < INCIDENCE_MAX:
            problems.append('%s is %s degrees (expected %g to %g)' % (
                key, a.get(key), INCIDENCE_MIN, INCIDENCE_MAX))
    p = a.get('PASS', '').upper()
    if p not in ('ASCENDING', 'DESCENDING'):
        problems.append('PASS is "%s"' % a.get('PASS', ''))
    else:
        op = pass_from_orbit(info)
        if op is None:
            problems.append('no usable orbit state vectors (needed to check PASS, '
                            'and by Terrain-Correction)')
        elif op != p:
            problems.append('PASS is %s but the orbit state vectors say %s' % (p, op))
        for src, hp in pass_hints(ds):
            if hp != p:
                problems.append('PASS is %s but %s says %s' % (p, src, hp))
    if problems and ds.get('convert'):
        problems.append('read from the PALSAR-2 format copy in %s/' % P2_DIR)
    elif problems and ds['mission'] == 'ALOS4':
        problems.append('this is what the ALOS-4 reader bug in SNAP 14.0.0-14.0.2 looks '
                        'like (GitHub senbox-org/microwave-toolbox issue #154); a SNAP '
                        'release with the fix (PR #155) is needed, or leave out '
                        '--native-alos4 to convert the data to PALSAR-2 format first')
    return problems, skip


def scene_from_info(ds, info, scene=None):
    a = info['abs']
    t = parse_utc(a.get('first_line_time'))
    scene = dict(scene or {})
    scene.pop('path', None)
    scene.update({'dataset': ds['name'], 'folder': os.path.relpath(ds['path']),
                  'vol': ds['vol'],
                  'read_by_snap': (P2_DIR + '/' + ds['p3']['vol2'] + ' (converted to '
                                   'PALSAR-2 format with JAXA\'s P3toP2snap code)'
                                   if ds.get('convert') else ds['vol']),
                  'mission': ds['mission'], 'mode': ds['mode'], 'pols': ds['pols'],
                  'date': t.strftime('%Y%m%d') if t else None,
                  'first_line_time': a.get('first_line_time'),
                  'pass': a.get('PASS'),
                  'acquisition_mode': a.get('ACQUISITION_MODE')})
    if not scene['date']:
        raise Failure('cannot read the acquisition date (first_line_time "%s")' %
                      a.get('first_line_time'))
    return scene


def multilook_summary(info):
    a = info['abs']
    rs, az = to_float(a.get('range_spacing')), to_float(a.get('azimuth_spacing'))
    inc_n, inc_f = to_float(a.get('incidence_near')), to_float(a.get('incidence_far'))
    ground = None
    if rs and inc_n and inc_f:
        ground = rs / math.sin(math.radians(0.5 * (inc_n + inc_f)))
    return {'range_looks': to_float(a.get('range_looks')),
            'azimuth_looks': to_float(a.get('azimuth_looks')),
            'slant_range_spacing_m': rs,
            'azimuth_spacing_m': az,
            'ground_range_spacing_m': ground,
            'incidence_near_deg': inc_n,
            'incidence_far_deg': inc_f,
            'size_pixels': [info['ncols'], info['nrows']]}


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def new_staging(path, name):
    # the process id in the name keeps a gpt left running by a killed run
    # from writing into this run's folder
    staging = os.path.join(path, '%s%s.%d' % (PARTIAL, name, os.getpid()))
    remove_path(staging)
    os.makedirs(staging)
    return staging


def clean_partials(path, log):
    for n in sorted(os.listdir(path)):
        if n.startswith(PARTIAL):
            log.write('removing leftover %s\n' % n)
            remove_path(os.path.join(path, n))


def move_product(staging, path, name):
    """Move name.dim/.data from the staging folder into place (.dim last:
    a .dim that exists means the product is complete)."""
    dst_dim, dst_data = dim_of(path, name), os.path.join(path, name + '.data')
    remove_path(dst_dim)
    remove_path(dst_data)
    os.rename(os.path.join(staging, name + '.data'), dst_data)
    os.rename(os.path.join(staging, name + '.dim'), dst_dim)
    remove_path(staging)


def remove_product(path, name):
    remove_path(dim_of(path, name))             # .dim first: then it counts as missing
    remove_path(os.path.join(path, name + '.data'))


def run_probe(ds, ctx, log, vol_path):
    staging = new_staging(ds['path'], 'probe')
    try:
        target = os.path.join(staging, 'probe.dim')
        graph = os.path.join(staging, 'probe_graph.xml')
        write_text(graph, graph_probe(vol_path, target))
        rc = run_logged([ctx['gpt'], graph] + gpt_extra(ctx), log)
        if rc != 0:
            raise Failure('reading the product failed (gpt exit code %d; see %s)' %
                          (rc, LOG_NAME))
        if not os.path.isfile(target):
            raise Failure('reading the product failed (no output; see %s)' % LOG_NAME)
        return read_dim(target)
    finally:
        remove_path(staging)


def run_step(ds, step, src, ctx, log):
    path, name = ds['path'], step['name']
    staging = new_staging(path, name)
    target = os.path.join(staging, name + '.dim')
    if step['kind'] == 'graph':
        # kept in the staging folder (it holds full paths); processing.json
        # records its parameters
        graph = os.path.join(staging, GRAPH_NAME)
        write_text(graph, graph_cal_mtx_mlk(src, target, ds['mode'], ctx['settings']))
        cmd = [ctx['gpt'], graph] + gpt_extra(ctx)
    else:
        cmd = ([ctx['gpt'], step['op'], '-Ssource=' + src, '-t', target] +
               ['-P%s=%s' % kv for kv in step['params']] + gpt_extra(ctx))
    rc = run_logged(cmd, log)
    if rc != 0:
        remove_path(staging)
        raise Failure('%s: gpt exited with code %d (see %s)' % (name, rc, LOG_NAME))
    problem = verify_product(target, step['bands'])
    if problem:
        remove_path(staging)
        raise Failure('%s: %s' % (name, problem))
    if step.get('box'):
        try:
            n = rezero_after_filter(target, src, ds['mode'], step['box'] // 2)
        except BaseException:
            remove_path(staging)
            raise
        log.write('%s: %d pixels next to no-data set back to no-data\n' % (name, n))
    move_product(staging, path, name)


# ---------------------------------------------------------------------------
# Export: ENVI files, NaN no-data, edge trim, RGB + PNG
# ---------------------------------------------------------------------------
def dilate(mask, r):
    """Square (2r+1) x (2r+1) binary dilation, without scipy."""
    if r <= 0:
        return mask
    tmp = mask.copy()
    for d in range(1, r + 1):
        tmp[:, d:] |= mask[:, :-d]
        tmp[:, :-d] |= mask[:, d:]
    out = tmp.copy()
    for d in range(1, r + 1):
        out[d:, :] |= tmp[:-d, :]
        out[:-d, :] |= tmp[d:, :]
    return out


def trim_radius(st, scene):
    """Terrain-Correction's bilinear resampling mixes no-data into output
    pixels within one input pixel of it: that many output pixels (1 or 2).
    (The box filters' mixing is undone in their own steps.)"""
    if not st['edge_trim']:
        return 0
    ml = (scene or {}).get('multilook') or {}
    g = max(ml.get('ground_range_spacing_m') or 0, ml.get('azimuth_spacing_m') or 0)
    if g <= 0:
        return 2                                # input pixel size unknown
    return max(1, int(math.ceil(g / float(st['pixel_spacing_m']))))


def load_band(path, w, h, big_endian=False):
    if os.path.getsize(path) != w * h * 4:
        raise Failure('%s has %d bytes, expected %d (%d x %d float32)' % (
            os.path.basename(path), os.path.getsize(path), w * h * 4, w, h))
    a = np.fromfile(path, dtype='>f4' if big_endian else '<f4', count=w * h)
    return a.reshape(h, w).astype(np.float32)


def save_band(arr, path, big_endian=False):
    tmp = path + '.tmp'
    arr.astype('>f4' if big_endian else '<f4').tofile(tmp)
    os.replace(tmp, path)


def band_img(dim, info, band):
    href = info['bands'][band]['file']
    p = os.path.join(os.path.dirname(dim), href)
    return p[:-4] + '.img' if p.endswith('.hdr') else p


def no_data_mask(dim, mode):
    """True where any diagonal element is 0 or not finite (SNAP's no-data)."""
    info = read_dim(dim)
    w, h = info['ncols'], info['nrows']
    mask = np.zeros((h, w), dtype=bool)
    for b in DIAG[mode]:
        x = load_band(band_img(dim, info, b), w, h, big_endian=True)
        mask |= ~np.isfinite(x) | (x == 0)
    return mask


def rezero_after_filter(new_dim, src_dim, mode, r):
    """SNAP's box filter averages no-data (0) into the pixels around it. Set
    every pixel whose filter window held no-data in the filter's input back
    to 0, in all matrix bands of the new product. Returns how many."""
    mask = dilate(no_data_mask(src_dim, mode), r)
    info = read_dim(new_dim)
    w, h = info['ncols'], info['nrows']
    if mask.shape != (h, w):
        raise Failure('box filter changed the image size')
    for b in BANDS[mode]:
        f = band_img(new_dim, info, b)
        x = load_band(f, w, h, big_endian=True)
        x[mask] = 0
        save_band(x, f, big_endian=True)
    return int(np.count_nonzero(mask))


def obs_mode(vol):
    """Observation mode code from a VOL file name (HBQ, FBD, FWD, ...)."""
    if vol.startswith('VOL-ALOS4'):
        return vol[22:25]
    parts = vol.split('-')                       # VOL-ALOS2...-yymmdd-HBQR1.1__A
    return parts[3][:3] if len(parts) > 3 else '?'


def png_labels(prov, warp_ref=None):
    """(title, caption) for the PNG previews, from an export's processing.json.
    The title is the original product ID (for reproducibility); the caption
    is the attribution, acquisition, mode, incidence range and grid.
    raster_plot.py shows the caption after "(R,G,B) = (...) ©"."""
    sc, st = prov.get('scene', {}), prov.get('settings', {})
    vol = sc.get('vol', '')
    title = vol[4:] if vol.startswith('VOL-') else (vol or 'unknown product')
    t = parse_utc(sc.get('first_line_time'))
    when = t.strftime('%Y-%m-%d %H:%M UTC') if t else sc.get('date', '?')
    year = t.strftime('%Y') if t else ''
    # kept short: raster_plot.py's fixed font fits about 115 characters,
    # "(R,G,B) = (T22, T33, T11) ©" included
    direction = {'ASCENDING': 'asc.', 'DESCENDING': 'desc.'}.get(
        (sc.get('pass') or '').upper(), '?')
    parts = ['JAXA %s' % year if year else 'JAXA',
             '%s, %s' % (when, direction),
             '%s %s-pol' % (obs_mode(vol), sc.get('mode', '?'))]
    ml = sc.get('multilook') or {}
    if ml.get('incidence_near_deg') and ml.get('incidence_far_deg'):
        parts.append('inc. %.0f-%.0f°' % (ml['incidence_near_deg'],
                                                ml['incidence_far_deg']))
    if warp_ref:
        parts.append('on %s grid' % warp_ref.split('_')[0])   # the reference's date
    else:
        parts.append('%s, %g m' % (st.get('map_projection', '?'),
                                   float(st.get('pixel_spacing_m') or 0)))
    return title, ' | '.join(parts)


def rgb_files(folder, name):
    rgb = os.path.join(folder, 'rgb')
    stack = os.path.join(rgb, name + '_rgb.bin')
    return rgb, stack, stack + '_1_2_3_rgb.png'


def rgb_ok(folder, name, title, caption):
    """True if the RGB stack and its PNG exist, with these labels."""
    rgb, stack, png = rgb_files(folder, name)
    if not all(os.path.isfile(f) for f in (stack, stack[:-4] + '.hdr', png)):
        return False
    for fn, want in (('title_string.txt', title), ('copyright_string.txt', caption)):
        try:
            with open(os.path.join(rgb, fn), encoding='utf-8') as f:
                if f.read().strip() != want:
                    return False
        except OSError:
            return False
    return True


def bands_ok(folder, mode, extra=()):
    """True if every matrix band (.bin + .hdr, of the size its header gives)
    and the extra files exist in folder."""
    for f in extra:
        if not os.path.isfile(os.path.join(folder, f)):
            return False
    for b in BANDS[mode]:
        f, hdr = os.path.join(folder, b + '.bin'), os.path.join(folder, b + '.hdr')
        if not (os.path.isfile(f) and os.path.isfile(hdr)):
            return False
        try:
            h = read_envi_header(hdr)
            if os.path.getsize(f) != int(h['samples']) * int(h['lines']) * 4:
                return False
        except (OSError, KeyError, ValueError):
            return False
    return True


def make_rgb(folder, name, mode, ctx, log, title, caption):
    """rgb/<name>_rgb.bin (+ .hdr) and its PNG, from the matrix bands in
    folder, labelled with title and caption. Built in a staging folder and
    swapped in when complete. Returns the PNG path."""
    for n in os.listdir(folder):                 # left by a stopped run
        if n.startswith(PARTIAL):
            remove_path(os.path.join(folder, n))
    final_rgb, _, final_png = rgb_files(folder, name)
    rgb = os.path.join(folder, '%srgb.%d' % (PARTIAL, os.getpid()))
    os.makedirs(rgb)
    py, tools, env = ctx['python'], ctx['tools'], ctx['env']
    stack = name + '_rgb.bin'

    def run(cmd, what):
        if run_logged(cmd, log, cwd=rgb, env=env) != 0:
            raise Failure('preview: %s failed (see %s)' % (what, LOG_NAME))

    try:
        _make_rgb(folder, rgb, stack, mode, py, tools, run, title, caption)
    except BaseException:
        remove_path(rgb)
        raise
    remove_path(final_rgb)
    os.rename(rgb, final_rgb)
    return final_png


def _make_rgb(folder, rgb, stack, mode, py, tools, run, title, caption):
    write_text(os.path.join(rgb, 'title_string.txt'), title + '\n')
    write_text(os.path.join(rgb, 'copyright_string.txt'), caption + '\n')

    if mode == 'quad':
        inputs = ['../T22.bin', '../T33.bin', '../T11.bin']
        names = ['T22', 'T33', 'T11']
    else:
        run([tools['convert_iq_to_cplx'], '../C12_real.bin', '../C12_imag.bin', 'C12.bin'],
            'convert_iq_to_cplx')
        run([tools['abs'], 'C12.bin'], 'abs')
        inputs = ['../C11.bin', '../C22.bin', 'C12.bin_abs.bin']
        names = ['C11', 'C22', '|C12|']
    run([py, tools['raster_stack.py']] + inputs + [stack], 'raster_stack.py')
    # A clean header, keeping the map information. misc.band_names() (used
    # by raster_plot.py) takes every line after "band names" as a band name,
    # so the band names go last, one per line.
    hdr0 = os.path.join(folder, names[0] + '.hdr')          # T22.hdr or C11.hdr
    first = read_envi_header(hdr0)
    lines = ['ENVI', 'samples = ' + first['samples'], 'lines = ' + first['lines'],
             'bands = 3', 'header offset = 0', 'file type = ENVI Standard',
             'data type = 4', 'interleave = bsq', 'byte order = 0']
    with open(hdr0) as f:
        lines += [ln.rstrip('\n') for ln in f
                  if ln.lower().startswith(('map info', 'coordinate system string'))]
    lines += ['band names = {' + names[0] + ',', names[1] + ',', names[2] + '}']
    write_text(os.path.join(rgb, stack[:-4] + '.hdr'), '\n'.join(lines) + '\n')
    run([py, tools['raster_plot.py'], stack, '1', '2', '3', '1'], 'raster_plot.py')
    png = os.path.join(rgb, stack + '_1_2_3_rgb.png')
    if not (os.path.isfile(os.path.join(rgb, stack)) and os.path.isfile(png)):
        raise Failure('preview: RGB stack or PNG not written (see %s)' % LOG_NAME)
    if mode == 'dual':                          # helper files for |C12|
        for n in ('C12.bin', 'C12.hdr', 'C12.bin.hdr', 'C12.bin_abs.bin', 'C12.bin_abs.hdr'):
            remove_path(os.path.join(rgb, n))


def nan_mask_and_save(folder, mode, w, h, trim=0, inc_path=None, inc_big_endian=True):
    """Set no-data (a zero or non-finite diagonal element, any non-finite
    element, and pixels within trim of those) to NaN in all matrix band files
    of folder; optionally divide by sin(incidence). Returns (valid, total)."""
    bands, diag = BANDS[mode], DIAG[mode]
    valid = np.ones((h, w), dtype=bool)
    for b in diag:
        x = load_band(os.path.join(folder, b + '.bin'), w, h)
        valid &= np.isfinite(x) & (x != 0)
        if np.count_nonzero(valid & (x < 0)) > 0.01 * max(1, np.count_nonzero(valid)):
            raise Failure('%s has many negative values: wrong byte order?' % b)
    for b in bands:                             # any non-finite element: invalid
        x = load_band(os.path.join(folder, b + '.bin'), w, h)
        valid &= np.isfinite(x)
    inv_sin = None
    if inc_path:
        inc = load_band(inc_path, w, h, big_endian=inc_big_endian)
        good = np.isfinite(inc) & (inc > 0) & (inc < 90)
        valid &= good
        inv_sin = np.zeros((h, w), dtype=np.float32)
        inv_sin[good] = 1.0 / np.sin(np.radians(inc[good]))
        del inc
    if trim > 0:
        valid = ~dilate(~valid, trim)
    for b in bands:
        f = os.path.join(folder, b + '.bin')
        x = load_band(f, w, h)
        if inv_sin is not None:
            x *= inv_sin
        x[~valid] = np.nan
        save_band(x, f)
    return int(np.count_nonzero(valid)), w * h


def export(ds, ctx, steps, scene, log):
    st, mode, path = ctx['settings'], ds['mode'], ds['path']
    name = '%s_%s_%s' % (scene['date'], ds['mission'], mode)
    out = os.path.join(path, name)
    if os.path.isdir(out):
        return out
    final = steps[-1]['name']
    staging = new_staging(path, name)
    try:
        src = os.path.join(path, final + '.data')
        info = read_dim(dim_of(path, final))
        w, h = info['ncols'], info['nrows']
        for b in BANDS[mode]:
            for ext in ('.img', '.hdr'):
                shutil.copy2(os.path.join(src, b + ext), os.path.join(staging, b + ext))
        # SNAP .img (byte order 1) -> .bin (byte order 0), in place
        rc = run_logged([ctx['python'], ctx['tools']['snap2psp_inplace.py']], log,
                        cwd=staging, env=ctx['env'])
        for b in BANDS[mode]:
            f, hdr = os.path.join(staging, b + '.bin'), os.path.join(staging, b + '.hdr')
            if (not os.path.isfile(f) or os.path.getsize(f) != w * h * 4 or
                    read_envi_header(hdr).get('byte order') != '0' or
                    os.path.exists(os.path.join(staging, b + '.img'))):
                raise Failure('export: snap2psp_inplace.py did not convert %s '
                              '(exit code %d; see %s)' % (b, rc, LOG_NAME))
        inc_path = None
        if st['tf_beta0_correction']:
            inc_path = os.path.join(path, TC_STEP + '.data', INC_BAND + '.img')
            if not os.path.isfile(inc_path):
                raise Failure('export: %s not found (needed by --tf-beta0-correction)' %
                              os.path.relpath(inc_path, path))
        trim = trim_radius(st, scene)
        n_valid, n_all = nan_mask_and_save(staging, mode, w, h, trim, inc_path)
        if n_valid == 0:
            raise Failure('export: no valid pixels left')
        prov = {'script': 'alos4_process.py ' + SCRIPT_VERSION,
                'made': now_str(),
                'snap_version': vstr(ctx['snap_version']),
                # identifies the data: an export made again from the same final
                # product holds the same values (the warp step relies on this)
                'from_product': {'name': final,
                                 'modified_ns': os.stat(dim_of(path, final)).st_mtime_ns},
                'settings': st, 'scene': scene,
                'steps': [{'name': s['name'], 'what': s['what'],
                           'operator': s.get('op', 'graph: Read, Calibration, '
                                             'Polarimetric-Matrices, Multilook, Write'),
                           'parameters': dict(s.get('params', []))} for s in steps],
                'export': {'bands': BANDS[mode], 'format': 'ENVI float32, byte order 0',
                           'no_data': 'NaN', 'size_pixels': [w, h],
                           'edge_trim_pixels': trim,
                           'tf_beta0_correction_applied': bool(inc_path),
                           'valid_pixels': n_valid, 'all_pixels': n_all,
                           'rgb_bands': RGB_BANDS[mode],
                           'png': os.path.relpath(rgb_files(staging, name)[2], staging)}}
        title, caption = png_labels(prov)
        make_rgb(staging, name, mode, ctx, log, title, caption)
        write_json(os.path.join(staging, 'processing.json'), prov)
        os.rename(staging, out)
    except BaseException:
        remove_path(staging)
        raise
    return out


def export_state(folder, mode):
    """'complete'; 'rgb' (bands fine, RGB preview / PNG missing or with old
    labels); or 'incomplete' (a matrix band or processing.json missing)."""
    if not bands_ok(folder, mode, ['processing.json']):
        return 'incomplete'
    prov = read_json(os.path.join(folder, 'processing.json'))
    if prov is None:
        return 'incomplete'
    title, caption = png_labels(prov)
    return 'complete' if rgb_ok(folder, os.path.basename(folder), title, caption) else 'rgb'


def repair_rgb(folder, mode, ctx, log):
    prov = read_json(os.path.join(folder, 'processing.json'))
    title, caption = png_labels(prov)
    make_rgb(folder, os.path.basename(folder), mode, ctx, log, title, caption)


# ---------------------------------------------------------------------------
# Clean-up
# ---------------------------------------------------------------------------
def remove_outputs(path):
    """Everything this script made in a dataset folder (except its log)."""
    removed = []
    for n in STEP_NAMES:
        if os.path.lexists(dim_of(path, n)) or os.path.lexists(os.path.join(path, n + '.data')):
            remove_product(path, n)
            removed.append(n)
    for d in export_dirs(path):
        remove_path(d)
        removed.append(os.path.basename(d))
    for n in sorted(os.listdir(path)):
        if n.startswith(PARTIAL) or n in (SETTINGS_NAME, SCENE_NAME, GRAPH_NAME, P2_DIR):
            remove_path(os.path.join(path, n))
            removed.append(n)
    return removed


def delete_intermediates(path, steps, st):
    """Delete everything the remaining work no longer needs: every step
    product older than the newest complete one (04_TC is kept while the beta0
    correction still needs it), and the PALSAR-2 format copy of ALOS-4 data
    once 01_CalMtxMlk (or anything later) exists. Safe at any point: whatever
    the next step reads is kept, and so is the final product (the export is
    made from it). Also removes half-deleted leftovers (a .data folder
    without its .dim). Returns what was removed."""
    keep = kept_steps(steps, st)
    names = [s['name'] for s in steps]
    have = [i for i, n in enumerate(names) if step_exists(path, n)]
    removed = []
    for i, n in enumerate(names):
        data = os.path.join(path, n + '.data')
        if i in have:
            if i < have[-1] and n not in keep:
                remove_product(path, n)
                removed.append(n)
        elif os.path.lexists(data):            # .data without .dim: incomplete
            remove_path(data)
            removed.append(n + '.data')
    if have and os.path.isdir(os.path.join(path, P2_DIR)):
        remove_path(os.path.join(path, P2_DIR))
        removed.append(P2_DIR)
    return removed


def prune(ds, steps, st, ctx, log):
    """delete_intermediates() when --delete-intermediates is on."""
    if not ctx['delete_intermediates']:
        return []
    removed = delete_intermediates(ds['path'], steps, st)
    if removed:
        log.write('deleted (no longer needed): %s\n' % ', '.join(removed))
        say('[%s] deleted (no longer needed): %s' % (ds['name'], ', '.join(removed)))
    return removed


# ---------------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------------
def process_dataset(ds, ctx):
    t0 = time.time()
    label, path, st = ds['name'], ds['path'], ctx['settings']
    res = {'name': label, 'status': 'failed', 'message': '', 'export': None}
    try:
        log = open(os.path.join(path, LOG_NAME), 'a', buffering=1)
    except OSError as e:
        res['message'] = 'cannot write in the dataset folder (%s)' % e
        return res
    try:
        log.write('\n' + '=' * 78 + '\n[%s] alos4_process.py %s  %s %s-pol\n' % (
            now_str(), SCRIPT_VERSION, ds['mission'], ds['mode']))
        log.write('settings: %s\n' % json.dumps(st, sort_keys=True))
        if ctx['redo']:
            removed = remove_outputs(path)
            if removed:
                log.write('--redo: removed %s\n' % ', '.join(removed))
                say('[%s] --redo: removed %d earlier outputs' % (label, len(removed)))
        clean_partials(path, log)
        steps = plan_steps(ds['mode'], st)
        check_settings(path, st, log, ds['mission'])
        scene = read_json(os.path.join(path, SCENE_NAME))

        done = [d for d in export_dirs(path)
                if os.path.basename(d).split('_')[1:] == [ds['mission'], ds['mode']]]
        state = export_state(done[0], ds['mode']) if done else 'missing'
        if state == 'incomplete':
            # matrix bands or processing.json missing: made again from the
            # final product (or, if that is gone too, from the newest step left)
            say('[%s] %s is incomplete: making the export again' % (
                label, os.path.basename(done[0])))
            log.write('export incomplete: removing %s\n' % os.path.basename(done[0]))
            remove_path(done[0])
        if state in ('complete', 'rgb'):
            res['export'] = done[0]
            if state == 'rgb':
                say('[%s] RGB preview / PNG missing or out of date: making only that' % label)
                repair_rgb(done[0], ds['mode'], ctx, log)
                res['message'] = 'RGB preview / PNG made again'
            else:
                res['message'] = 'already done'
        else:
            # an existing first product must pass the checks before it is used
            s1 = steps[0]['name']
            if step_exists(path, s1):
                problems, skip = check_metadata(read_dim(dim_of(path, s1)), ds, 's1')
                if skip:
                    raise Skip(skip)
                if problems:
                    log.write('existing %s fails the checks: %s\n' % (s1, '; '.join(problems)))
                    say('[%s] existing %s fails the metadata checks: removing it and '
                        'everything made from it' % (label, s1))
                    for n in STEP_NAMES:
                        remove_product(path, n)
                    scene = None
                elif not scene or not scene.get('multilook'):
                    info = read_dim(dim_of(path, s1))
                    scene = scene_from_info(ds, info, scene)
                    scene['multilook'] = multilook_summary(info)
                    write_json(os.path.join(path, SCENE_NAME), scene)

            prune(ds, steps, st, ctx, log)      # e.g. left over from a run without it
            todo = steps_to_run(path, steps, st)
            snap_vol = os.path.join(path, ds['vol'])
            if todo and todo[0] == 0:
                if ds.get('convert'):
                    if not p2_complete(ds, os.path.join(path, P2_DIR)):
                        say('[%s] converting PALSAR-3 to PALSAR-2 format (JAXA P3toP2snap) '
                            'into %s/ ...' % (label, P2_DIR))
                    t1 = time.time()
                    snap_vol = p3_to_p2(ds, log)
                    log.write('PALSAR-2 format copy ready in %s\n' % fmt_time(time.time() - t1))
                say('[%s] checking metadata (probe)' % label)
                info = run_probe(ds, ctx, log, snap_vol)
                problems, skip = check_metadata(info, ds, 'probe')
                if skip:
                    raise Skip(skip)
                if problems:
                    raise Failure('metadata checks failed: ' + '; '.join(problems))
                scene = scene_from_info(ds, info)
                write_json(os.path.join(path, SCENE_NAME), scene)
                log.write('probe passed: %s %s %s\n' % (scene['date'], scene['pass'],
                                                        scene['acquisition_mode']))
            if not scene:
                # first product deleted earlier: the metadata of any later one will do
                latest = [s['name'] for s in steps if step_exists(path, s['name'])]
                if not latest:
                    raise Failure('no product to read the metadata from')
                scene = scene_from_info(ds, read_dim(dim_of(path, latest[-1])))
                write_json(os.path.join(path, SCENE_NAME), scene)

            for i in todo:
                s = steps[i]
                src = snap_vol if i == 0 else dim_of(path, steps[i - 1]['name'])
                say('[%s] %s  %s ...' % (label, s['name'], s['what']))
                t1 = time.time()
                run_step(ds, s, src, ctx, log)
                if i == 0:
                    info = read_dim(dim_of(path, s['name']))
                    problems, skip = check_metadata(info, ds, 's1')
                    if skip or problems:
                        remove_product(path, s['name'])
                        if skip:
                            raise Skip(skip)
                        raise Failure('%s fails the metadata checks: %s' % (
                            s['name'], '; '.join(problems)))
                    scene = scene_from_info(ds, info, scene)
                    scene['multilook'] = ml = multilook_summary(info)
                    write_json(os.path.join(path, SCENE_NAME), scene)
                    looks = ('looks: %g range x %g azimuth, pixel about %.1f m (ground range) '
                             'x %.1f m (azimuth)' % (ml['range_looks'] or 0,
                                                     ml['azimuth_looks'] or 0,
                                                     ml['ground_range_spacing_m'] or 0,
                                                     ml['azimuth_spacing_m'] or 0))
                    log.write(looks + '\n')
                    say('[%s] %s' % (label, looks))
                say('[%s] %s  done in %s' % (label, s['name'], fmt_time(time.time() - t1)))
                prune(ds, steps, st, ctx, log)  # the step before is no longer needed

            say('[%s] export: ENVI files, NaN no-data, RGB preview ...' % label)
            res['export'] = export(ds, ctx, steps, scene, log)
            res['message'] = 'done in %s' % fmt_time(time.time() - t0)
        res['status'] = 'done'
        prune(ds, steps, st, ctx, log)
    except Skip as e:
        res['status'], res['message'] = 'skipped', str(e)
    except Failure as e:
        res['message'] = str(e)
    except KeyboardInterrupt:
        res['message'] = 'interrupted'
        log.write('[%s] interrupted\n' % now_str())
        raise
    except Exception as e:
        res['message'] = 'unexpected error: %s (traceback in %s)' % (e, LOG_NAME)
        log.write(traceback.format_exc())
    finally:
        log.write('[%s] %s: %s\n' % (now_str(), res['status'], res['message']))
        log.close()
    return res


# ---------------------------------------------------------------------------
# Parallel workers
# ---------------------------------------------------------------------------
def _worker_init(lock):
    global PRINT_LOCK
    PRINT_LOCK = lock
    # The main process handles Ctrl-C and hang-up, and stops the workers
    # with SIGTERM, which stops each worker's gpt (see run_logged).
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def on_term(signum, frame):
        raise SystemExit(1)
    signal.signal(signal.SIGTERM, on_term)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def catch_stop_signals():
    """kill (SIGTERM) or a closed terminal (SIGHUP) stop the run like Ctrl-C,
    so gpt is stopped too (it runs in its own process group)."""
    signal.signal(signal.SIGTERM, _interrupt)
    if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:     # not under nohup
        signal.signal(signal.SIGHUP, _interrupt)


def _worker(job):
    ds, ctx = job
    try:
        return process_dataset(ds, ctx)
    except Exception as e:                      # SystemExit (stop) must end the worker
        return {'name': ds['name'], 'status': 'failed',
                'message': 'unexpected error: %s' % e, 'export': None}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='ALOS-2 / ALOS-4 Level 1.1 dual- and quad-pol processing with SNAP gpt '
                    '(run in the folder that holds the dataset folders).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-j', '--jobs', type=int, default=1,
                   help='datasets processed at the same time (each runs its own gpt)')
    p.add_argument('--rg-looks', type=int, default=4, help='range looks')
    p.add_argument('--az-looks', type=int, default=None,
                   help='fixed azimuth looks (default: SNAP derives them per scene '
                        'for square ground pixels)')
    p.add_argument('--no-tf', action='store_true', help='skip Terrain-Flattening')
    p.add_argument('--tf-beta0-correction', action='store_true',
                   help='divide the exported matrix by sin(ellipsoid incidence angle), '
                        'undoing the sigma0/beta0 offset of Terrain-Flattening on ALOS data')
    p.add_argument('--no-pre-box', action='store_true', help='skip the box filter before TC')
    p.add_argument('--pre-box-size', type=int, default=3, help='box filter size before TC')
    p.add_argument('--no-post-box', action='store_true', help='skip the box filter after TC')
    p.add_argument('--post-box-size', type=int, default=3, help='box filter size after TC')
    p.add_argument('--pixel-spacing', type=float, default=20.0,
                   help='output pixel size (m)')
    p.add_argument('--crs', default='EPSG:3005', help='output map projection (BC Albers)')
    p.add_argument('--no-edge-trim', action='store_true',
                   help='keep the output pixels Terrain-Correction mixed with no-data '
                        '(1 or 2 pixels from it; biased low)')
    p.add_argument('--delete-intermediates', action='store_true',
                   help='delete each intermediate product as soon as the next step no '
                        'longer needs it (the PALSAR-2 format copy of ALOS-4 data once '
                        '01_CalMtxMlk is made, each step product once the next one is); '
                        'the final product is kept')
    p.add_argument('--delete-intermediates-only', action='store_true',
                   help='only delete what the remaining steps no longer need, in every '
                        'dataset (finished or not); process nothing')
    p.add_argument('--redo', action='store_true',
                   help="delete this script's outputs for each dataset and process it again")
    p.add_argument('--dry-run', action='store_true',
                   help='list the datasets and what would run, then stop')
    p.add_argument('--gpt', default=None, help='path to SNAP gpt (default: found automatically)')
    p.add_argument('--native-alos4', action='store_true',
                   help='read ALOS-4 data with SNAP\'s own ALOS-4 reader instead of converting '
                        'it to PALSAR-2 format first (only once SNAP has the fix for GitHub '
                        'senbox-org/microwave-toolbox issue #154)')
    p.add_argument('--no-warp', action='store_true',
                   help='do not put the finished outputs on one common grid (in ./warp/) '
                        'at the end')
    p.add_argument('--warp-only', action='store_true',
                   help='only do the warp step (on the outputs finished so far); '
                        'process nothing')
    p.add_argument('--warp-ref', default=None,
                   help='warp reference: a date (YYYYMMDD) or a name such as '
                        '20250622_ALOS4_quad (default: the earliest)')
    p.add_argument('--warp-bilinear', action='store_true',
                   help='warp with bilinear instead of nearest-neighbour resampling')
    a = p.parse_args()
    if a.jobs < 1:
        p.error('--jobs must be 1 or more')
    if a.rg_looks < 1 or (a.az_looks is not None and a.az_looks < 1):
        p.error('looks must be 1 or more')
    for flag, n in (('--pre-box-size', a.pre_box_size), ('--post-box-size', a.post_box_size)):
        if n < 3 or n % 2 == 0:
            p.error('%s must be odd and at least 3' % flag)
    if a.pixel_spacing <= 0:
        p.error('--pixel-spacing must be positive')
    if a.tf_beta0_correction and a.no_tf:
        p.error('--tf-beta0-correction only applies with Terrain-Flattening (drop --no-tf)')
    if a.redo and a.delete_intermediates_only:
        p.error('--redo and --delete-intermediates-only do not go together')
    if a.no_warp and a.warp_only:
        p.error('--no-warp and --warp-only do not go together')
    return a


# ---------------------------------------------------------------------------
# Warp: all finished exports onto one common grid (was alos4_warp.py)
# ---------------------------------------------------------------------------
WARP_DIR = 'warp'
WARP_LOG = 'alos4_warp.log'
GIF_DELAY = 111                                 # hundredths of a second per frame


def warp_collect():
    """Finished exports of every dataset folder here, sorted by name."""
    datasets, _ = find_datasets('.')
    entries, unfinished = [], []
    for ds in datasets:
        dirs = export_dirs(ds['path'])
        if not dirs:
            unfinished.append(ds['name'])
        for d in dirs:
            name = os.path.basename(d)
            date, mission, mode = name.split('_')
            missing = [b for b in BANDS[mode] if not os.path.isfile(os.path.join(d, b + '.bin'))]
            if missing:
                print('WARNING: %s/%s lacks %s: left out of the warp' % (
                    ds['name'], name, ', '.join(missing)))
                continue
            entries.append({'name': name, 'date': date, 'mission': mission, 'mode': mode,
                            'dataset': ds['name'], 'dir': d,
                            'ref_band': os.path.join(d, DIAG[mode][0] + '.bin')})
    entries.sort(key=lambda e: e['name'])
    return entries, unfinished


def warp_grid(path):
    from osgeo import gdal
    gdal.UseExceptions()
    try:
        g = gdal.Open(path)
    except RuntimeError as e:
        raise Failure('GDAL cannot open %s (%s)' % (path, e))
    return g.GetGeoTransform(), g.GetProjection(), g.RasterXSize, g.RasterYSize


def warp_same_grid(gt, wkt, ref_gt, ref_wkt):
    """True if a raster's pixels coincide with the reference's pixels."""
    from osgeo import osr
    if not wkt or not ref_wkt:
        return False
    a, b = osr.SpatialReference(), osr.SpatialReference()
    a.ImportFromWkt(wkt)
    b.ImportFromWkt(ref_wkt)
    if not a.IsSame(b):
        return False
    px, py = ref_gt[1], ref_gt[5]
    if gt[2] != 0 or gt[4] != 0 or ref_gt[2] != 0 or ref_gt[4] != 0:
        return False
    if abs(gt[1] - px) > 1e-6 * abs(px) or abs(gt[5] - py) > 1e-6 * abs(py):
        return False
    for off, step in ((gt[0] - ref_gt[0], px), (gt[3] - ref_gt[3], py)):
        k = off / step
        if abs(k - round(k)) > 1e-3:
            return False
    return True


def warp_outputs(ref_choice, bilinear, tools, env):
    """Copy the reference export and resample every other finished export
    onto its grid, in ./warp/. Returns a list of (name, problem)."""
    print('\nwarp: putting every finished output on one common grid in %s/' % WARP_DIR)
    entries, unfinished = warp_collect()
    if unfinished:
        print('left out (no finished output: not processed, failed or skipped): %s' %
              ', '.join(unfinished))
    if not entries:
        print('warp: no finished outputs yet: nothing to do')
        return []
    names = [e['name'] for e in entries]
    dups = sorted(set(n for n in names if names.count(n) > 1))
    if dups:
        return [('warp', 'more than one dataset is called %s (folders: %s); one observation '
                 'per date is assumed: move one of them aside' % (', '.join(dups), ', '.join(
                     e['dataset'] for e in entries if e['name'] in dups)))]
    if ref_choice:
        m = [e for e in entries if ref_choice in (e['name'], e['date'])]
        if len(m) != 1:
            return [('warp', '--warp-ref %s matches %d datasets (%s); use a full name' % (
                ref_choice, len(m), ', '.join(e['name'] for e in m) or 'none'))]
        ref = m[0]
    else:
        ref = entries[0]

    out = os.path.abspath(WARP_DIR)
    rt = os.path.join(out, 'reference.txt')
    if os.path.isfile(rt):
        old = open(rt).read().split()
        if old and old[0] != ref['name']:
            # e.g. an earlier date was processed since: every output must move
            print('the reference was %s, now it is %s: making %s/ again from the start '
                  '(use --warp-ref %s to keep the old reference)' % (
                      old[0], ref['name'], WARP_DIR, old[0]))
            remove_path(out)
    os.makedirs(out, exist_ok=True)
    if not os.path.isfile(rt):
        write_text(rt, '%s\n%s\n' % (ref['name'], os.path.relpath(ref['dir'])))
    for n in os.listdir(out):
        if n.startswith(PARTIAL):
            remove_path(os.path.join(out, n))
    project = os.path.join(misc.pd, 'raster_project_onto.py')
    ctx = {'python': sys.executable, 'tools': tools, 'env': env}

    ref_gt, ref_wkt, ref_w, ref_h = warp_grid(ref['ref_band'])
    print('reference: %s (%s), %d x %d pixels of %g m' % (
        ref['name'], ref['dataset'], ref_w, ref_h, abs(ref_gt[1])))
    if not ref_wkt:
        print('WARNING: GDAL finds no map projection for %s' % ref['ref_band'])
    off_grid = []
    for e in entries:
        gt, wkt, _, _ = warp_grid(e['ref_band'])
        if not warp_same_grid(gt, wkt, ref_gt, ref_wkt):
            off_grid.append(e['name'])
    if off_grid:
        print('WARNING: not on the reference grid (values will be resampled%s): %s' % (
            '' if bilinear else '; consider --warp-bilinear', ', '.join(off_grid)))

    log = open(os.path.join(out, WARP_LOG), 'a', buffering=1)
    log.write('\n[%s] warp, reference %s\n' % (now_str(), ref['name']))
    resampling = 'bilinear' if bilinear else 'nearest neighbour'
    done, failed = [], []
    for e in entries:
        target = os.path.join(out, e['name'])
        prov = read_json(os.path.join(e['dir'], 'processing.json')) or {}
        # what identifies the source data (older exports: when it was made)
        source_key = json.dumps(prov.get('from_product') or prov.get('made'), sort_keys=True)
        title, caption = png_labels(prov, warp_ref=ref['name'])
        if os.path.isdir(target):
            w = read_json(os.path.join(target, 'warp.json')) or {}
            same = ((w.get('source_key') or json.dumps(w.get('source_made'))) == source_key
                    and w.get('resampling') == resampling)
            if same and bands_ok(target, e['mode'], ['warp.json']):
                try:
                    if rgb_ok(target, e['name'], title, caption):
                        print('  %-24s already done' % e['name'])
                    else:
                        print('  %-24s RGB preview / PNG missing or out of date: making '
                              'only that' % e['name'])
                        make_rgb(target, e['name'], e['mode'], ctx, log, title, caption)
                    done.append(e)
                except (Failure, OSError) as x:
                    failed.append((e['name'], str(x)))
                    print('  %-24s FAILED: %s' % (e['name'], x))
                continue
            # processed again since, other resampling asked for, or bands missing
            print('  %-24s %s: making it again' % (
                e['name'], 'out of date' if not same else 'incomplete'))
            remove_path(target)
        print('  %-24s %s ...' % (e['name'], 'copying (reference)' if e is ref else
                                  'resampling (%s)' % resampling), flush=True)
        staging = os.path.join(out, '%s%s.%d' % (PARTIAL, e['name'], os.getpid()))
        remove_path(staging)
        os.makedirs(staging)
        try:
            bands = BANDS[e['mode']]
            if e is ref:
                for b in bands:
                    for ext in ('.bin', '.hdr'):
                        shutil.copy2(os.path.join(e['dir'], b + ext),
                                     os.path.join(staging, b + ext))
            else:
                cmds = [[sys.executable, project, os.path.join(e['dir'], b + '.bin'),
                         ref['ref_band'], os.path.join(staging, b + '.bin')] +
                        ([] if bilinear else ['nearest']) for b in bands]
                with ThreadPoolExecutor(max_workers=4) as pool:
                    rcs = list(pool.map(lambda c: run_logged(c, log, env=env), cmds))
                bad = [b for b, rc in zip(bands, rcs) if rc != 0 or not os.path.isfile(
                    os.path.join(staging, b + '.bin'))]
                if bad:
                    raise Failure('raster_project_onto.py failed for %s (see %s/%s)' % (
                        ', '.join(bad), WARP_DIR, WARP_LOG))
            n_valid, n_all = nan_mask_and_save(staging, e['mode'], ref_w, ref_h)
            png = make_rgb(staging, e['name'], e['mode'], ctx, log, title, caption)
            write_json(os.path.join(staging, 'warp.json'), {
                'script': 'alos4_process.py ' + SCRIPT_VERSION, 'made': now_str(),
                'source': os.path.relpath(e['dir']), 'source_key': source_key,
                'reference': ref['name'], 'reference_folder': os.path.relpath(ref['dir']),
                'copied_not_resampled': e is ref, 'resampling': resampling,
                'on_reference_grid': e['name'] not in off_grid,
                'size_pixels': [ref_w, ref_h], 'valid_pixels': n_valid,
                'all_pixels': n_all, 'no_data': 'NaN',
                'png': os.path.relpath(png, staging)})
            os.rename(staging, target)
            done.append(e)
            if n_valid == 0:
                print('  %-24s WARNING: no overlap with the reference' % e['name'])
        except (Failure, OSError) as x:
            remove_path(staging)
            failed.append((e['name'], str(x)))
            print('  %-24s FAILED: %s' % (e['name'], x))
        except BaseException:
            remove_path(staging)
            raise

    # PNG copies and one animation per polarisation mode (the colours differ)
    # (each made only if missing or older than what it is made from)
    def stale(f, sources):
        return (not os.path.isfile(f) or
                any(os.path.getmtime(s) > os.path.getmtime(f) for s in sources))

    for e in done:
        png = rgb_files(os.path.join(out, e['name']), e['name'])[2]
        copy = os.path.join(out, e['name'] + '.png')
        if os.path.isfile(png) and stale(copy, [png]):
            shutil.copy2(png, copy)
    convert = shutil.which('convert') or shutil.which('magick')
    for mode in ('quad', 'dual'):
        pngs = [os.path.join(out, e['name'] + '.png') for e in done
                if e['mode'] == mode and os.path.isfile(os.path.join(out, e['name'] + '.png'))]
        gif = os.path.join(out, 'sequence_%s.gif' % mode)
        frames_txt = os.path.join(out, 'sequence_%s.txt' % mode)    # the frames, in order
        frames = '\n'.join(os.path.basename(p) for p in pngs) + '\n'
        if len(pngs) < 2:
            continue
        try:
            same_frames = open(frames_txt).read() == frames
        except OSError:
            same_frames = False
        if same_frames and not stale(gif, pngs):
            continue
        if not convert:
            print('ImageMagick (convert) not found: no %s' % os.path.basename(gif))
            continue
        tmp = os.path.join(out, '.tmp_sequence_%s.gif' % mode)
        if run_logged([convert, '-delay', str(GIF_DELAY)] + pngs + [tmp], log) == 0 \
                and os.path.isfile(tmp):
            os.replace(tmp, gif)
            write_text(frames_txt, frames)
            print('%s: %d frames' % (os.path.relpath(gif), len(pngs)))
        else:
            remove_path(tmp)
            print('making %s failed (see %s/%s)' % (os.path.basename(gif), WARP_DIR, WARP_LOG))
    log.close()
    print('warp: %d on the grid of %s in %s/%s' % (
        len(done), ref['name'], WARP_DIR, ', %d failed' % len(failed) if failed else ''))
    return failed


def run_warp(a):
    """The warp step with the tools it needs; returns a list of problems."""
    tools, env, missing = find_tools()
    missing = [m for m in missing if not m.startswith('sbo_inplace') and
               not m.endswith('snap2psp_inplace.py')]          # not used by the warp
    if not os.path.isfile(os.path.join(misc.pd, 'raster_project_onto.py')):
        missing.append(os.path.join(misc.pd, 'raster_project_onto.py'))
    if missing:
        return [('warp', 'missing tools: %s' % ', '.join(missing))]
    try:
        return warp_outputs(a.warp_ref, a.warp_bilinear, tools, env)
    except Failure as e:
        return [('warp', str(e))]


def main():
    a = parse_args()
    st = settings_from_args(a)
    datasets, ignored = find_datasets('.')
    print('alos4_process.py %s in %s' % (SCRIPT_VERSION, os.getcwd()))
    if ignored:
        print('folders without a VOL-ALOS2/VOL-ALOS4 file (ignored): %s' % ', '.join(ignored))
    if not datasets:
        err('no dataset folders (with a VOL-ALOS2* or VOL-ALOS4* file inside) here')

    runnable, results = [], []
    for ds in datasets:
        reason = classify(ds, convert_alos4=not a.native_alos4)
        where = ('' if ds['path'] == os.path.abspath(ds['name'])
                 else '  (data in %s/%s)' % (ds['name'], ds['name']))
        if reason:
            results.append({'name': ds['name'], 'status': 'skipped', 'message': reason,
                            'export': None})
            print('  %-28s skipped: %s' % (ds['name'], reason))
        else:
            runnable.append(ds)
            print('  %-28s %s %s-pol (%s)%s%s' % (
                ds['name'], ds['mission'], ds['mode'], '+'.join(ds['pols']), where,
                ', %s mode, read via PALSAR-2 format' % ds['p3']['mode']
                if ds.get('convert') else ''))

    if a.warp_only:
        problems = run_warp(a)
        for n, m in problems:
            print('FAILED %s: %s' % (n, m))
        return 1 if problems else 0

    if a.delete_intermediates_only:
        # finished or not: only what the remaining steps no longer need goes
        # (do not run this while another run is processing the same folder)
        for ds in runnable:
            old = read_json(os.path.join(ds['path'], SETTINGS_NAME))
            if old is None:
                print('  %-28s not processed yet: left alone' % ds['name'])
                continue
            removed = delete_intermediates(ds['path'], plan_steps(ds['mode'], old), old)
            print('  %-28s %s' % (ds['name'], 'deleted ' + ', '.join(removed)
                                   if removed else 'nothing to delete'))
        return 0

    print('options: %s' % json.dumps(st, sort_keys=True))
    if a.dry_run:
        for ds in runnable:
            steps = plan_steps(ds['mode'], st)
            done = [d for d in export_dirs(ds['path'])
                    if os.path.basename(d).split('_')[1:] == [ds['mission'], ds['mode']]]
            state = export_state(done[0], ds['mode']) if done and not a.redo else 'missing'
            if state == 'complete':
                plan = 'finished (%s)' % os.path.basename(done[0])
            elif state == 'rgb':
                plan = 'make only the RGB preview / PNG of %s' % os.path.basename(done[0])
            else:
                todo = (list(range(len(steps))) if a.redo
                        else steps_to_run(ds['path'], steps, st))
                names = [steps[i]['name'] for i in todo]
                if (ds.get('convert') and todo and todo[0] == 0 and
                        (a.redo or not p2_complete(ds, os.path.join(ds['path'], P2_DIR)))):
                    names.insert(0, 'convert to PALSAR-2 format')
                plan = ' -> '.join(names + ['export'])
            print('  %-28s %s' % (ds['name'], plan))
        print('  then: %s' % ('no warp (--no-warp)' if a.no_warp else
                              'warp every finished output onto one grid in %s/' % WARP_DIR))
        return 0
    if not runnable:
        print('nothing to process')
        if a.no_warp:
            return 0
        problems = run_warp(a)
        for n, m in problems:
            print('FAILED %s: %s' % (n, m))
        return 1 if problems else 0

    gpt, version = pick_gpt(a.gpt)
    tools, env, missing = find_tools()
    if missing:
        err('missing tools needed for the export: %s. Compile the wps-research C++ '
            'tools (wps-research/cpp) and make sure they are on the PATH.' % ', '.join(missing))
    jobs = min(a.jobs, len(runnable))
    threads = max(1, (os.cpu_count() or 1) // jobs) if jobs > 1 else None
    ctx = {'gpt': gpt, 'snap_version': version, 'settings': st, 'tools': tools,
           'env': env, 'threads': threads, 'python': sys.executable,
           'redo': a.redo, 'delete_intermediates': a.delete_intermediates}
    print('SNAP gpt: %s (SNAP %s)' % (gpt, vstr(version)))
    print('processing %d dataset(s), %d at a time%s; gpt output goes to each folder\'s %s'
          % (len(runnable), jobs, (', %d threads each' % threads) if threads else '',
             LOG_NAME))
    t0 = time.time()
    interrupted = False
    catch_stop_signals()
    try:
        if jobs == 1:
            for ds in runnable:
                results.append(process_dataset(ds, ctx))
                r = results[-1]
                say('[%s] %s: %s' % (r['name'], r['status'], r['message']))
        else:
            lock = multiprocessing.Lock()
            global PRINT_LOCK
            PRINT_LOCK = lock
            pool = multiprocessing.Pool(jobs, initializer=_worker_init, initargs=(lock,))
            try:
                for r in pool.imap_unordered(_worker, [(ds, ctx) for ds in runnable]):
                    results.append(r)
                    say('[%s] %s: %s' % (r['name'], r['status'], r['message']))
                pool.close()
            except KeyboardInterrupt:
                pool.terminate()
                raise
            finally:
                pool.join()
    except KeyboardInterrupt:
        interrupted = True
        print('\ninterrupted: run the same command again to carry on')

    warp_problems = []
    if not interrupted and not a.no_warp:
        try:
            warp_problems = run_warp(a)
        except KeyboardInterrupt:
            interrupted = True
            print('\ninterrupted during the warp: run the same command again to carry on')

    print('\nsummary (%s):' % fmt_time(time.time() - t0))
    for status in ('done', 'skipped', 'failed'):
        rs = [r for r in results if r['status'] == status]
        if rs:
            print('  %s: %d' % (status, len(rs)))
            for r in sorted(rs, key=lambda r: r['name']):
                extra = ('  -> ' + os.path.relpath(r['export'])) if r.get('export') else ''
                print('    %-28s %s%s' % (r['name'], r['message'], extra))
    if a.no_warp:
        print('  warp: off (--no-warp)')
    elif not interrupted:
        print('  warp: %s' % ('%d problem(s)' % len(warp_problems) if warp_problems
                              else 'done (%s/)' % WARP_DIR))
        for n, m in warp_problems:
            print('    %-28s %s' % (n, m))
    if interrupted:
        return 130
    return 1 if (any(r['status'] == 'failed' for r in results) or warp_problems) else 0


if __name__ == '__main__':
    sys.exit(main())
