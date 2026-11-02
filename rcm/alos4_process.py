#!/usr/bin/env python3
"""
20261101 alos_polsar_process.py

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

Processing chain (all outputs go inside the dataset's own folder)
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
    A dataset fails if any of these is wrong. SNAP 14.0.2 misreads ALOS-4
    metadata and still reports success (GitHub senbox-org/microwave-toolbox
    issue #154, "ALOS-4 (PALSAR-3) CEOS reader misreads the leader"; fixed
    on master by PR #155 on Oct 30 2026, in no release yet). These checks
    are what catch it:
      - MISSION in the metadata matches the VOL file (ALOS2 / ALOS4)
      - range and azimuth pixel spacing greater than 0
      - slant range to the first pixel between 400 and 2500 km
      - near and far incidence angles between 5 and 75 degrees
      - PASS (ASCENDING / DESCENDING) agrees with the direction the
        satellite was moving (orbit state vectors), with the ALOS-2 product
        name's last letter (A / D), and with the .kml or summary.txt if
        either of those spells the direction out
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
    Safe to stop at any time (Ctrl-C) and run again with the same options.
    Each step writes into a hidden ".partial_<step>" folder inside the
    dataset folder and is moved into place only after gpt succeeds and the
    output checks out, so any output that exists is complete. Leftover
    ".partial_*" folders are deleted on the next run.
    The options a dataset was processed with are saved in
    alos_polsar_settings.json. If a later run asks for different processing
    options, that dataset fails with a message instead of mixing outputs;
    --redo deletes this script's outputs for each dataset and starts again
    (if a --redo run is stopped, carry on without --redo; datasets it had
    not reached yet keep their earlier outputs).

Usage (run from the folder that holds the dataset folders)
    python3 alos_polsar_process.py                    # all defaults
    python3 alos_polsar_process.py --dry-run          # list datasets, plan
    python3 alos_polsar_process.py -j 4               # 4 datasets at a time
    python3 alos_polsar_process.py --no-tf --no-pre-box
    python3 alos_polsar_process.py --delete-intermediates
    python3 alos_polsar_process.py --delete-intermediates-only
    python3 alos_polsar_process.py --redo
    python3 alos_polsar_process.py -h                 # all options

Needs (as the older rcm scripts do): wps-research/py on the same checkout
(misc.py, snap2psp_inplace.py, raster_stack.py, raster_plot.py) and the
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
import contextlib
import signal
import argparse
import datetime
import threading
import traceback
import subprocess
import multiprocessing
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

sep = os.path.sep
my_path = os.path.dirname(os.path.abspath(__file__)) + sep
sys.path.append(my_path + '..' + sep + 'py')
import misc                                   # wps-research/py/misc.py
from misc import find_snap, err
import numpy as np

SCRIPT_VERSION = '20261101'

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


def classify(ds):
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
    return None


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
          'edge_trim': not a.no_edge_trim}
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


def check_settings(path, st, log):
    """Refuse to mix outputs made with different options."""
    f = os.path.join(path, SETTINGS_NAME)
    old = read_json(f)
    if old is None:
        if outputs_present(path):
            raise Failure('outputs exist but %s is missing, so the options they '
                          'were made with are unknown; use --redo' % SETTINGS_NAME)
    else:
        diff = [k for k in st if k not in EXPORT_ONLY_KEYS and old.get(k) != st[k]]
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
    out = ['<graph id="alos_polsar_process">', '  <version>1.0</version>']
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
    if mission != ds['mission']:
        problems.append('MISSION is "%s" but the VOL file is %s' % (mission, ds['mission']))
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
    if problems and ds['mission'] == 'ALOS4':
        problems.append('this is what the ALOS-4 reader bug in SNAP 14.0.0-14.0.2 looks '
                        'like (GitHub senbox-org/microwave-toolbox issue #154); a SNAP '
                        'release with the fix (PR #155) is needed')
    return problems, skip


def scene_from_info(ds, info, scene=None):
    a = info['abs']
    t = parse_utc(a.get('first_line_time'))
    scene = dict(scene or {})
    scene.pop('path', None)
    scene.update({'dataset': ds['name'], 'folder': os.path.relpath(ds['path']),
                  'vol': ds['vol'],
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
    staging = os.path.join(path, PARTIAL + name)
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


def run_probe(ds, ctx, log):
    staging = new_staging(ds['path'], 'probe')
    try:
        target = os.path.join(staging, 'probe.dim')
        graph = os.path.join(staging, 'probe_graph.xml')
        write_text(graph, graph_probe(os.path.join(ds['path'], ds['vol']), target))
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


def make_rgb(folder, name, mode, ctx, log):
    """rgb/<name>_rgb.bin (+ .hdr) and its PNG, from the matrix bands in folder.
    Returns the PNG path."""
    rgb = os.path.join(folder, 'rgb')
    remove_path(rgb)
    os.makedirs(rgb)
    py, tools, env = ctx['python'], ctx['tools'], ctx['env']
    stack = name + '_rgb.bin'

    def run(cmd, what):
        if run_logged(cmd, log, cwd=rgb, env=env) != 0:
            raise Failure('preview: %s failed (see %s)' % (what, LOG_NAME))

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
    # raster_stack.py takes every header line after "band names" for a band
    # name; write a clean header instead, keeping the map information
    hdr0 = os.path.join(folder, names[0] + '.hdr')          # T22.hdr or C11.hdr
    first = read_envi_header(hdr0)
    lines = ['ENVI', 'samples = ' + first['samples'], 'lines = ' + first['lines'],
             'bands = 3', 'header offset = 0', 'file type = ENVI Standard',
             'data type = 4', 'interleave = bsq', 'byte order = 0',
             'band names = {' + ', '.join(names) + '}']
    with open(hdr0) as f:
        lines += [ln.rstrip('\n') for ln in f
                  if ln.lower().startswith(('map info', 'coordinate system string'))]
    write_text(os.path.join(rgb, stack[:-4] + '.hdr'), '\n'.join(lines) + '\n')
    run([py, tools['raster_plot.py'], stack, '1', '2', '3', '1'], 'raster_plot.py')
    png = os.path.join(rgb, stack + '_1_2_3_rgb.png')
    if not (os.path.isfile(os.path.join(rgb, stack)) and os.path.isfile(png)):
        raise Failure('preview: RGB stack or PNG not written (see %s)' % LOG_NAME)
    if mode == 'dual':                          # helper files for |C12|
        for n in ('C12.bin', 'C12.hdr', 'C12.bin.hdr', 'C12.bin_abs.bin', 'C12.bin_abs.hdr'):
            remove_path(os.path.join(rgb, n))
    return png


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
        png = make_rgb(staging, name, mode, ctx, log)
        prov = {'script': 'alos_polsar_process.py ' + SCRIPT_VERSION,
                'made': now_str(),
                'snap_version': vstr(ctx['snap_version']),
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
                           'png': os.path.relpath(png, staging)}}
        write_json(os.path.join(staging, 'processing.json'), prov)
        os.rename(staging, out)
    except BaseException:
        remove_path(staging)
        raise
    return out


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
        if n.startswith(PARTIAL) or n in (SETTINGS_NAME, SCENE_NAME, GRAPH_NAME):
            remove_path(os.path.join(path, n))
            removed.append(n)
    return removed


def delete_intermediates(path, steps, st):
    keep = kept_steps(steps, st)
    removed = []
    for s in steps:
        n = s['name']
        if n not in keep and (os.path.lexists(dim_of(path, n)) or
                              os.path.lexists(os.path.join(path, n + '.data'))):
            remove_product(path, n)
            removed.append(n)
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
        log.write('\n' + '=' * 78 + '\n[%s] alos_polsar_process.py %s  %s %s-pol\n' % (
            now_str(), SCRIPT_VERSION, ds['mission'], ds['mode']))
        log.write('settings: %s\n' % json.dumps(st, sort_keys=True))
        if ctx['redo']:
            removed = remove_outputs(path)
            if removed:
                log.write('--redo: removed %s\n' % ', '.join(removed))
                say('[%s] --redo: removed %d earlier outputs' % (label, len(removed)))
        clean_partials(path, log)
        steps = plan_steps(ds['mode'], st)
        check_settings(path, st, log)
        scene = read_json(os.path.join(path, SCENE_NAME))

        done = [d for d in export_dirs(path)
                if os.path.basename(d).split('_')[1:] == [ds['mission'], ds['mode']]]
        if done:
            res['export'] = done[0]
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

            todo = steps_to_run(path, steps, st)
            if todo and todo[0] == 0:
                say('[%s] checking metadata (probe)' % label)
                info = run_probe(ds, ctx, log)
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
                src = (os.path.join(path, ds['vol']) if i == 0
                       else dim_of(path, steps[i - 1]['name']))
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

            say('[%s] export: ENVI files, NaN no-data, RGB preview ...' % label)
            res['export'] = export(ds, ctx, steps, scene, log)
            res['message'] = 'done in %s' % fmt_time(time.time() - t0)
        res['status'] = 'done'
        if ctx['delete_intermediates']:
            removed = delete_intermediates(path, steps, st)
            if removed:
                log.write('deleted intermediates: %s\n' % ', '.join(removed))
                res['message'] += '; deleted %s' % ', '.join(removed)
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
                   help='after a dataset is finished, delete its intermediate SNAP '
                        'products (the final one is kept)')
    p.add_argument('--delete-intermediates-only', action='store_true',
                   help='only delete intermediate products of finished datasets; process nothing')
    p.add_argument('--redo', action='store_true',
                   help="delete this script's outputs for each dataset and process it again")
    p.add_argument('--dry-run', action='store_true',
                   help='list the datasets and what would run, then stop')
    p.add_argument('--gpt', default=None, help='path to SNAP gpt (default: found automatically)')
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
    return a


def main():
    a = parse_args()
    st = settings_from_args(a)
    datasets, ignored = find_datasets('.')
    print('alos_polsar_process.py %s in %s' % (SCRIPT_VERSION, os.getcwd()))
    if ignored:
        print('folders without a VOL-ALOS2/VOL-ALOS4 file (ignored): %s' % ', '.join(ignored))
    if not datasets:
        err('no dataset folders (with a VOL-ALOS2* or VOL-ALOS4* file inside) here')

    runnable, results = [], []
    for ds in datasets:
        reason = classify(ds)
        where = ('' if ds['path'] == os.path.abspath(ds['name'])
                 else '  (data in %s/%s)' % (ds['name'], ds['name']))
        if reason:
            results.append({'name': ds['name'], 'status': 'skipped', 'message': reason,
                            'export': None})
            print('  %-28s skipped: %s' % (ds['name'], reason))
        else:
            runnable.append(ds)
            print('  %-28s %s %s-pol (%s)%s' % (ds['name'], ds['mission'], ds['mode'],
                                                 '+'.join(ds['pols']), where))

    if a.delete_intermediates_only:
        for ds in runnable:
            old = read_json(os.path.join(ds['path'], SETTINGS_NAME))
            done = [d for d in export_dirs(ds['path'])
                    if os.path.basename(d).split('_')[1:] == [ds['mission'], ds['mode']]]
            if old is None or not done:
                print('  %-28s not finished: left alone' % ds['name'])
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
            if done and not a.redo:
                plan = 'finished (%s)' % os.path.basename(done[0])
            else:
                todo = (list(range(len(steps))) if a.redo
                        else steps_to_run(ds['path'], steps, st))
                plan = ' -> '.join(steps[i]['name'] for i in todo) or 'nothing'
                plan += ' -> export'
            print('  %-28s %s' % (ds['name'], plan))
        return 0
    if not runnable:
        print('nothing to process')
        return 0

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

    print('\nsummary (%s):' % fmt_time(time.time() - t0))
    for status in ('done', 'skipped', 'failed'):
        rs = [r for r in results if r['status'] == status]
        if rs:
            print('  %s: %d' % (status, len(rs)))
            for r in sorted(rs, key=lambda r: r['name']):
                extra = ('  -> ' + os.path.relpath(r['export'])) if r.get('export') else ''
                print('    %-28s %s%s' % (r['name'], r['message'], extra))
    if interrupted:
        return 130
    return 1 if any(r['status'] == 'failed' for r in results) else 0


if __name__ == '__main__':
    sys.exit(main())
