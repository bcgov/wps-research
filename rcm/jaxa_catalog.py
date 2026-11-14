#!/usr/bin/env python3
"""jaxa_catalogue.py -- catalogue ALOS-2 / ALOS-4 zip products on all mounted volumes

USAGE
  python3 jaxa_catalogue.py [--merge | --inspect] [--verify] [--include-alos1] [--no-qgis] [PATH ...]

  python3 jaxa_catalogue.py                  # scan all volumes -> CSV -> shapefiles -> QGIS
  python3 jaxa_catalogue.py /data /mnt/usb   # scan only these paths (one worker each)
  python3 jaxa_catalogue.py --no-qgis        # everything except opening QGIS at the end
  python3 jaxa_catalogue.py --merge          # combine every catalogue CSV below here -> CSV -> shapefiles
  python3 jaxa_catalogue.py --inspect DIR    # full structure report on the zips in DIR (writes nothing)
  python3 jaxa_catalogue.py --verify         # also CRC-check one copy of every scene (slow: reads all data)
  python3 jaxa_catalogue.py --merge --verify # merge, verifying the copies on this machine
  python3 jaxa_catalogue.py --include-alos1  # also catalogue ALOS-1 products (left out by default)
  mv JAXA_products_catalogue.csv old.csv     # forces a fresh scan on the next run

ALOS-1 (left out unless --include-alos1)
  ALOS-1 products (2006-2011) are not catalogued by default. They are
  recognised by file name (ALOS1 / ALOS-1, or the ALOS-1 product codes ALPSRP,
  ALAV2, ALPSM) during the search, and by the satellite in their metadata
  after reading. --merge also drops ALOS-1 rows found in older catalogues.
  --include-alos1 keeps them all, in the search, the merge and the outputs.

SCENES AND REPRESENTATIVE COPIES
  Copies of a scene are grouped by scene_key (the zip name without extension
  or part number), so damaged and undamaged copies of a scene are grouped
  together. One copy per scene is the representative (is_representative =
  yes), chosen in this order: passed --verify, then undamaged status, then
  the size most undamaged copies share, then a copy on this machine.
  The shapefiles and JAXA_products_summary.csv use representatives only
  (one per scene); the catalogue CSV keeps every copy. Per row:
    scene_state          ok | ok, verified | ok, not verified |
                         NEEDS RE-DOWNLOAD: no undamaged copy found |
                         ...; SIZE CONFLICT (undamaged copies differ in size)
    n_copies             copies of this scene found (all machines when merged)
    n_copies_ok / _damaged   how many of them are undamaged / damaged
    representative_path / _host   the copy chosen for this scene
  Scenes needing attention are listed at the end of each run, with every copy.

VERIFY (--verify)
  The normal checks cover the zip structure (end records, directory, every
  file inside complete). --verify also reads every byte of every file in the
  zip and compares its CRC-32 and size with the zip directory, which catches
  corrupted data. It checks the best undamaged copy of each scene that is on
  this machine; if that copy fails, the next copy is checked, and so on.
  Result per row in verify_status (passed: ... / FAILED: ...). Slow: reads
  all the data (several GB per ALOS-4 scene). Results are kept in the CSV and
  carried through --merge, so run --verify on each machine for its copies;
  a merge then prefers copies that passed anywhere. With an existing CSV,
  --verify checks the copies it lists and updates the CSV (no new search).

STATUS COLUMN (one per product; damaged zips are also listed at the end of a run)
  ok                  metadata and footprint read
  no_footprint        JAXA metadata read, but no corner coordinates found
  not_jaxa_product    readable zip whose name matches *ALOS*, but no JAXA files in it
  damaged_recovered   broken zip structure; metadata read from the intact files
  damaged_unreadable  broken zip; nothing could be read
  error               unexpected error while reading
  error says what is wrong. For damaged zips, diagnosis holds a structural
  report: size and first bytes of each part, the zip end records and whether
  they are consistent, a scan of the zip's file headers from the start
  (where the data stops, inside which file) and a CONCLUSION (e.g.
  INCOMPLETE: a part is missing or cut short).

MERGE MODE (--merge)
  For combining catalogues made on different machines / file systems: copy
  each machine's JAXA_products_catalogue.csv into its own subfolder of one
  folder (any depth, e.g. heartofgold/, laptop/usb1/), cd to that folder
  and run with --merge.
  * Exits with an error if JAXA_products_catalogue.csv already exists in
    the present folder, since that is where the merged CSV is written.
  * Searches no volumes and opens no zips: it finds every
    JAXA_products_catalogue.csv below the present folder, joins their rows
    into a new JAXA_products_catalogue.csv here, then makes all the usual
    shapefiles (1-4 below) from it and opens QGIS.
  * Rows that are exact repeats (the same CSV copied twice) are kept once.
    n_copies is recounted over all machines. source_csv says which CSV each
    row came from; catalogue_host says which machine scanned it (CSVs from
    older versions of this script have no catalogue_host or zip_size).
  * CSVs from the first version (lt/rt/lb/rb corner columns) are converted
    to the current NW/NE/SW/SE corners.
  * Also writes 5. JAXA_products_summary.csv: the merged rows grouped by
    equivalence class, classes with the most unique datasets first, with
    these columns in front:
      class_id             zero-padded class number (as in item 3)
      class_shapefile      JAXA_class_members/<class_id>_members.shp
      class_unique_scenes  datasets in the class, copies counted once
      scene_n_parts        number of zip parts of this dataset (1 = single zip)
      scene_n_copies       copies of this dataset found in total
    Rows without corners come last, with an empty class_id.

FILE SIZES (normal and --merge runs)
  One CSV row per dataset (a multipart zip is one row): zip_size is the
  total of its parts, part_sizes lists each part (same order as all_parts).
  At the end of the CSV stage two totals over all data files are printed:
    space used   every file at every location (machine + path) counted
    unique data  identical files (same name and size) counted once

WHAT IT DOES
  Stage 1 (skipped if JAXA_products_catalogue.csv already exists):
    a) Lists every mounted volume (Linux: /proc/self/mounts, macOS: `mount`)
       and runs one worker per volume:
         find <volume> -xdev \\( -name '*ALOS*.zip' -o ... \\) -type f
       -xdev keeps each worker on its own volume, so no volume is searched
       twice. Each worker returns a list whose entries are an absolute path
       (single zip) or a list of absolute paths (the parts of a multipart zip,
       which must all be in the same folder). The lists are then joined.
    b) A work queue of 4 workers reads the metadata of each entry straight
       from the zip(s), without unzipping, and writes the CSV.
  Stage 2 (always runs, from the CSV):
    1. JAXA_footprints_unique.shp   one polygon per unique set of 4 exact
                                    corners
    2. JAXA_footprints_classes.shp  one polygon per equivalence class:
                                    polygons are linked when EVERY corner is
                                    within CORNER_THRESHOLD_KM of the matching
                                    corner of the other (NW to NW, NE to NE,
                                    SW to SW, SE to SE); classes are the
                                    transitive closure of those links.
                                    Attribute CLASS_ID is the zero-padded
                                    number used in item 3.
    3. JAXA_class_members/<CLASS_ID>_members.shp
                                    one shapefile per equivalence class, with
                                    one polygon per product in that class.
                                    CLASS_ID always has at least two leading
                                    zeros (e.g. 001, 002 ... or 00001 ...).
                                    Classes are numbered largest first.
    4. JAXA_scene_centres.shp       one point per product with a scene
                                    centre, including products whose corners
                                    could not be found (HAS_CORNER = 0);
                                    CLASS_ID links points to item 2.
    5. JAXA_products_summary.csv (--merge only): one row per scene, grouped by
                                    area then class, with area_id, area_name,
                                    scene_site, class_id, ... in front.
    6. JAXA_areas.shp               TOP LEVEL: one polygon per place. Classes
                                    over the same place are joined whatever
                                    their mode, satellite or direction (they
                                    overlap by >= AREA_OVERLAP_FRACTION of the
                                    smaller footprint). NAME comes first, then
                                    NAME_STAT (resolved / several names /
                                    unresolved / manual), N_SCENES, N_DATES,
                                    N_MODES, MODES (e.g. FBD,HBQ), SAT_MODES
                                    (e.g. ALOS2 FBD,ALOS4 HBQ), SATS, ORB_DIRS,
                                    CLASS_IDS, FIRST/LAST_ACQ, YEARS, N_COPIES,
                                    N_REDOWNLD, HOSTS, SITES, NAME_HINT, DATES,
                                    then DATE001, DATE002, ...: one field per
                                    scene, same text as the DATES pieces, in
                                    time order (e.g. 2016-07-11 FBD ALOS2 A).
       JAXA_areas.csv               the same, with the full list of dates
                                    and modes per area.
  NAME, AREA_ID and CLASS_ID are also on the unique, class, member and centre
  shapefiles; member shapefiles also have SITE (the scene's own site folders).
    7. JAXA_areas_unresolved.shp    only the areas still without a name (same
                                    fields as 6); not written if none.
    8. JAXA_areas/<AREA_ID>_<NAME>.shp
                                    one shapefile per area (same fields as 6),
                                    e.g. JAXA_areas/0009_Elephant_Hill.shp;
                                    the FILE field of 6 points to it.
       JAXA_areas/<AREA_ID>_<NAME>.kmz
                                    the same area for Google Earth: outline,
                                    and a pop-up with its attributes and
                                    every acquisition date with its mode.
  Finally QGIS is opened with 6 (areas), 7 (unresolved areas) and 4 (scene
  centres).

SITE NAMES
  Taken from folder names "<year>_<name>" in the paths of all copies of a
  scene on all machines, e.g. JAXA_COLLECT/2021_sparks_lake -> Sparks Lake
  (2021). Generic folders (2025_JAXA_DATA, 2023_EODMS_data_from_Hao, ...) are
  skipped via SITE_STOPWORDS. Folders named otherwise can be mapped to a
  site name in FOLDER_SITE_NAMES (e.g. PALSAR_SF -> SanFransisco). An area with no site folder is named
  "UNRESOLVED (<its most common folder name>)"; one with several names lists
  them all ("several names"). Name places yourself in MANUAL_SITES (name,
  lat, lon) at the top of the file; a manual name overrides folder names.

MULTIPART ZIPS (parts must be in the same folder)
  name.z01, name.z02, ..., name.zip      spanned archive (Info-ZIP / WinZip)
  name.zip.001, name.zip.002, ...        one zip cut into byte chunks (7-Zip;
  name.zip.0001, name.zip.0002, ...      any number of digits, as JAXA uses),
                                         or complete zips numbered the same way
  name_part1.zip, name_part2.zip, ...    separate complete zips (see PART_REGEX)

METADATA READ (scalar values only; no vectors, orbit state vectors or
calibration tables)
  summary*.txt    every key="value" line (scene times, off-nadir angle,
                  scene and product IDs, image/frame corners for Level 1.5+,
                  ...). ALOS-2 names it summary.txt, ALOS-4
                  summary-<scene ID>.txt.
  workreport*     same key="value" format, if present
  LED-* (CEOS)    data set summary record (ALOS-2 4096-byte and ALOS-4
                  4680-byte layouts): incidence angle at scene centre,
                  scene centre time, orbit direction, wavelength, platform,
                  sensor mode, orbit number, nominal off-nadir angle
  *.kml           scene footprint (needed for Level 1.1, whose summary has
                  no corners): gx:LatLonQuad, LinearRing or LatLonBox
  *.xml           leaf values that occur once (repeated tags are vectors)
  file names      polarizations (IMG-HH-..., IMG-HV-...), scene and product IDs
  The first CSV columns are standardised (dates/times, polarizations,
  incidence angle, corners, ...); all raw values follow, sorted by name.

CORNERS (tried in this order; the first that works is used)
  1. summary file corner keys (Level 1.5+)
  2. KML footprint in the zip
  3. CEOS image files (IMG-XX-...): each line record holds the lat/lon of
     its first, middle and last pixel (Level 1.1: bytes 193-216, Level 1.5:
     133-156). The first and last records give the corners. For a
     compressed image file this means decompressing it in memory up to its
     last record (a few seconds per GB; nothing is written to disk).
  4. GeoTIFF image in the zip (via GDAL; single zips only)
  Corners are labelled geographically, whatever their source: the two most
  northerly points are NW/NE (west = smaller longitude), the other two SW/SE.
  So footprints from different sources compare corner for corner.
  corner_source in the CSV says where they came from.

SCENE CENTRE (first that works)
  summary centre keys, leader file (Level 1.5+), middle image record,
  GeoTIFF centre, mean of the corners. center_source says which.

COPIES
  n_copies counts files with the same name and size found anywhere (e.g. a
  zip copied to a second folder). Each copy keeps its own CSV row.

SETTINGS
  Adjustable constants are right below the imports (CORNER_THRESHOLD_KM etc.).

REQUIREMENTS
  Python 3.8+, POSIX `find`. Stage 2 needs GDAL's Python bindings (osgeo).
  If those were built for NumPy 1.x but NumPy 2 is installed (GDAL then prints
  "A module that was compiled using NumPy 1.x cannot be run in NumPy 2..."),
  the script fixes it at startup: it removes a NumPy 2 in your user folder
  that hides the NumPy GDAL was built with, or installs numpy<2 (pip --user,
  ready-made wheels only), then restarts. If neither is possible it skips
  GDAL's NumPy part, which it does not use, so no error is ever printed.
"""
import argparse
import collections
import csv
import glob
import io
import math
import multiprocessing as mp
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# GDAL's NumPy part (osgeo.gdal_array) is never used here. When it cannot load
# (GDAL built for NumPy 1.x, NumPy 2 installed) GDAL prints a long NumPy error
# every time it switches on exceptions; marking it unavailable silences that.
# Set by fix_gdal_numpy() below; inherited by worker processes.
if os.environ.get('JAXA_NO_GDAL_ARRAY') == '1':
    sys.modules['osgeo.gdal_array'] = None

# ---------------------------------------------------------------- settings --
CORNER_THRESHOLD_KM = 1.0     # max distance between matching corners (item 2)
CSV_FILE = 'JAXA_products_catalogue.csv'
SHP_UNIQUE = 'JAXA_footprints_unique.shp'
SHP_CLASSES = 'JAXA_footprints_classes.shp'
SHP_CENTRES = 'JAXA_scene_centres.shp'
SUMMARY_FILE = 'JAXA_products_summary.csv'     # written by --merge
SHP_AREAS = 'JAXA_areas.shp'                   # top level: one polygon per place
SHP_UNRESOLVED = 'JAXA_areas_unresolved.shp'   # only the areas without a name
AREAS_DIR = 'JAXA_areas'                       # one shapefile per area: <AREA_ID>_<NAME>.shp
AREAS_CSV = 'JAXA_areas.csv'                   # areas with their full date lists
AREA_OVERLAP_FRACTION = 0.5   # classes join one area when they overlap by at least
                              # this fraction of the smaller footprint (any mode)

# Site (AOI) names come from folder names "<year>_<site name>" in the paths of
# all copies of a scene, e.g. 2021_sparks_lake -> "Sparks Lake (2021)".
SITE_FOLDER_REGEX = re.compile(r'^((?:19|20)\d\d)[_\- ]+([A-Za-z].*)$')
# folder names containing any of these words are not site names
SITE_STOPWORDS = {
    'jaxa', 'data', 'backup', 'website', 'eodms', 'duplicate', 'apsar', 'collect',
    'archive', 'misc', 'other', 'tmp', 'temp', 'zip', 'zips', 'download', 'downloads',
    'alos', 'alos1', 'alos2', 'alos3', 'alos4', 'palsar', 'palsar2', 'palsar3', 'from',
    'batch', 'catalog', 'catalogue', 'important', 'bak', 'test', 'sample', 'samples',
    'order', 'orders', 'ftp', 'pub', 'sar', 'esa', 'polsarpro', 'processed', 'output',
}
# ALOS-1 (2006-2011) products are left out unless --include-alos1 is given.
# Recognised by file name (ALOS1 / ALOS-1, or JAXA's ALOS-1 product codes
# ALPSRP = PALSAR, ALAV2 = AVNIR-2, ALPSM = PRISM) and, after reading, by the
# satellite named in the metadata.
INCLUDE_ALOS1 = False                          # set by --include-alos1
ALOS1_NAME_RE = re.compile(r'(ALOS[-_ ]?1(?!\d)|ALPSRP|ALAV2|ALPSM)', re.I)
ALOS1_SATELLITES = {'ALOS', 'ALOS1', 'ALOS-1'}

# Folders that name a site without the "<year>_" pattern: folder name
# (case-insensitive) -> site name.
FOLDER_SITE_NAMES = {
    'PALSAR_SF': 'SanFransisco',
    'EODMS_ALOS2_FP6_DSC': 'EODMS_ALOS2_FP6_DSC',
    'EODMS_ALOS2_FP6_ASC': 'EODMS_ALOS2_FP6_ASC',
}
# Names for places with no site folder: (name, latitude, longitude). An area
# whose polygon contains the point gets that name (overrides folder names).
MANUAL_SITES = [
    ("Mount Underwood", 49.358, -124.824),   # 2025 fire south of Port Alberni
    ("Elaho", 50.30, -123.74),               # 2015 Elaho Valley fire, 67 km W of Pemberton
    ("Ainslie Creek", 49.807, -121.505),     # largest 2026 fire near Boston Bar (Brunswick Complex)
    ("Adams Lake", 50.899, -119.424),        # Shuswap
    ("Donnie Creek", 57.813, -121.111),      # 2023 fire, NE BC
    ("Donnie Creek", 57.323, -120.950),      # same region, 2020-2022 data
]
CLASS_DIR = 'JAXA_class_members'
N_META_WORKERS = 4            # metadata work-queue size

# find -name patterns: single zips plus the parts of multipart zips
FIND_PATTERNS = ['*ALOS*.zip', '*ALOS*.z[0-9][0-9]', '*ALOS*.zip.[0-9]*']

# separate complete zips that form one product, e.g. X_part1.zip, X-part02.zip
PART_REGEX = re.compile(r'^(?P<base>.+?)[._-]?part[._-]?(?P<num>\d+)\.zip$', re.I)

# kernel / virtual file systems that never hold data files
SKIP_FS_TYPES = {
    'proc', 'sysfs', 'devtmpfs', 'devpts', 'cgroup', 'cgroup2', 'pstore',
    'securityfs', 'debugfs', 'tracefs', 'configfs', 'fusectl', 'mqueue',
    'hugetlbfs', 'bpf', 'binfmt_misc', 'autofs', 'rpc_pipefs', 'nsfs',
    'efivarfs', 'selinuxfs', 'devfs', 'squashfs', 'nullfs',
}
# mount points (and everything under them) that are never scanned
SKIP_MOUNT_PREFIXES = (
    '/proc', '/sys', '/dev', '/snap',
    '/System/Volumes/VM', '/System/Volumes/Preboot', '/System/Volumes/Update',
    '/System/Volumes/xarts', '/System/Volumes/iSCPreboot',
    '/System/Volumes/Hardware', '/Volumes/com.apple.TimeMachine.localsnapshots',
)

MAX_TEXT_MEMBER = 4 * 1024 * 1024   # skip metadata members bigger than this
LED_BYTES = 720 + 4096              # CEOS file descriptor + data set summary

# standardised columns, written first in the CSV
STD_COLUMNS = [
    'product', 'site_names', 'status', 'is_representative', 'scene_state', 'verify_status', 'satellite', 'scene_id', 'product_id', 'processing_level',
    'observation_mode', 'polarizations', 'look_direction', 'orbit_direction',
    'acq_start', 'acq_center', 'acq_end',
    'incidence_angle_center', 'off_nadir_angle',
    'center_lat', 'center_lon',
    'center_source',
    'nw_lat', 'nw_lon', 'ne_lat', 'ne_lon',
    'sw_lat', 'sw_lon', 'se_lat', 'se_lon', 'corner_source',
    'orbit_number', 'frame_number', 'pixel_spacing',
    'archive_type', 'n_parts', 'n_copies', 'n_copies_ok', 'n_copies_damaged',
    'scene_key', 'representative_path', 'representative_host', 'zip_size', 'part_sizes', 'zip_path', 'all_parts',
    'catalogue_host', 'source_csv', 'meta_files', 'error', 'diagnosis',
]
CORNER_KEYS = ['nw_lat', 'nw_lon', 'ne_lat', 'ne_lon',
               'sw_lat', 'sw_lon', 'se_lat', 'se_lon']
EARTH_RADIUS_KM = 6371.0088


def log(msg):
    print(msg, flush=True)


# =========================================================== stage 1a: find
def decode_mount_path(s):
    """/proc/mounts escapes space, tab, newline and backslash as \\ooo."""
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), s)


def list_volumes():
    """Mounted volumes worth scanning: [(mount point, fs type), ...]."""
    mounts = []
    if os.path.exists('/proc/self/mounts'):                       # Linux
        with open('/proc/self/mounts') as f:
            for line in f:
                p = line.split()
                if len(p) >= 3:
                    mounts.append((decode_mount_path(p[1]), p[2]))
    else:                                                         # macOS, BSD
        out = subprocess.run(['mount'], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL).stdout.decode('utf-8', 'replace')
        for line in out.splitlines():
            m = re.match(r'^(.+?) on (.+?)(?: type (\S+))? \((.*)\)\s*$', line)
            if m:
                fstype = m.group(3) or m.group(4).split(',')[0].strip()
                mounts.append((m.group(2), fstype))

    volumes, seen = [], set()
    for mnt, fstype in mounts:
        if fstype in SKIP_FS_TYPES:
            continue
        if any(mnt == p or mnt.startswith(p + '/') for p in SKIP_MOUNT_PREFIXES):
            continue
        if mnt in seen or not os.path.isdir(mnt):
            continue
        seen.add(mnt)
        volumes.append((mnt, fstype))
    return volumes


def group_parts(paths):
    """Turn found files into entries: a path (single zip) or a list of paths
    (parts of one multipart zip, in reading order)."""
    singles, spanned, split, parts = {}, {}, {}, {}
    for p in paths:
        d, name = os.path.split(p)
        m = re.match(r'^(.*)\.z(\d{2,})$', name, re.I)               # x.z01
        if m:
            spanned.setdefault((d, m.group(1)), {})[int(m.group(2))] = p
            continue
        m = re.match(r'^(.*\.zip)\.(\d{3,})$', name, re.I)           # x.zip.001
        if m:
            split.setdefault((d, m.group(1)), {})[int(m.group(2))] = p
            continue
        if name.lower().endswith('.zip'):
            m = PART_REGEX.match(name)                               # x_part1.zip
            if m:
                parts.setdefault((d, m.group('base')), {})[int(m.group('num'))] = p
            else:
                singles[p] = (d, name[:-4])

    entries = []
    spanned_last = {}
    for p, key in singles.items():
        if key in spanned:
            spanned_last[key] = p        # final .zip of a spanned set
        else:
            entries.append(p)
    for key, g in spanned.items():
        lst = [g[k] for k in sorted(g)]
        if key in spanned_last:
            lst.append(spanned_last[key])
        entries.append(lst)
    for g in split.values():
        entries.append([g[k] for k in sorted(g)])
    for g in parts.values():
        lst = [g[k] for k in sorted(g)]
        entries.append(lst if len(lst) > 1 else lst[0])
    return entries


def scan_volume(mount_point):
    """Worker: find ALOS zips on one volume. Returns (mount, entries, n_errors)."""
    cmd = ['find', mount_point, '-xdev', '(']
    for i, pat in enumerate(FIND_PATTERNS):
        if i:
            cmd.append('-o')
        cmd += ['-name', pat]
    cmd += [')', '-type', 'f', '-print']
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = proc.stdout.decode('utf-8', 'surrogateescape')
    paths = [os.path.abspath(x) for x in out.splitlines() if x]
    n_err = len(proc.stderr.splitlines())    # mostly "Permission denied"
    return mount_point, group_parts(paths), n_err


def dedupe_entries(entries):
    """Drop entries seen twice through different mount points (bind mounts,
    macOS /System/Volumes/Data firmlinks). Keeps the cleanest path."""
    best = {}
    for e in entries:
        p = e if isinstance(e, str) else e[0]
        try:
            st = os.stat(p)
            key = (st.st_dev, st.st_ino)
        except OSError:
            key = ('path', p)
        rank = (p.startswith('/System/Volumes/Data/'), len(p), p)
        if key not in best or rank < best[key][0]:
            best[key] = (rank, e)
    return [v[1] for v in best.values()]


def is_alos1_name(path):
    return bool(ALOS1_NAME_RE.search(os.path.basename(path)))


def is_alos1_row(r):
    """An ALOS-1 product, by its metadata (satellite) or its file name."""
    return (r.get('satellite', '').upper() in ALOS1_SATELLITES
            or is_alos1_name(r.get('zip_path', '')) or is_alos1_name(r.get('product', '')))


def drop_alos1(rows, where):
    """Remove ALOS-1 rows unless --include-alos1 was given."""
    if INCLUDE_ALOS1:
        return rows
    kept = [r for r in rows if not is_alos1_row(r)]
    if len(kept) < len(rows):
        log(f"  left out {len(rows) - len(kept)} ALOS-1 product(s) {where} "
            f"(use --include-alos1 to keep them)")
    return kept


def stage_scan(paths):
    if paths:
        volumes = [(os.path.abspath(p), 'given') for p in paths if os.path.isdir(p)]
        missing = [p for p in paths if not os.path.isdir(p)]
        for p in missing:
            log(f"[WARN] not a directory, skipped: {p}")
    else:
        volumes = list_volumes()
    if not volumes:
        log("[ERROR] no volumes to scan")
        return []

    log(f"Scanning {len(volumes)} volume(s), one worker each:")
    for mnt, fstype in volumes:
        log(f"  {mnt}  ({fstype})")
    log("")

    results, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=len(volumes)) as pool:
        futures = [pool.submit(scan_volume, mnt) for mnt, _ in volumes]
        for fut in as_completed(futures):
            mnt, entries, n_err = fut.result()
            n_multi = sum(1 for e in entries if not isinstance(e, str))
            note = f", {n_err} unreadable folder(s)" if n_err else ""
            log(f"  done {mnt}: {len(entries)} product(s), {n_multi} multipart"
                f"{note}  [{time.time() - t0:.0f}s]")
            results.append(entries)

    joined = [e for lst in results for e in lst]                 # join the lists
    if not INCLUDE_ALOS1:                                        # ALOS-1 by file name
        before = len(joined)
        joined = [e for e in joined
                  if not is_alos1_name(e if isinstance(e, str) else e[0])]
        if len(joined) < before:
            log(f"  left out {before - len(joined)} ALOS-1 product(s) by file name "
                f"(use --include-alos1 to keep them)")
    unique = dedupe_entries(joined)
    if len(unique) < len(joined):
        log(f"  removed {len(joined) - len(unique)} duplicate(s) seen via other mount points")
    log(f"\nFound {len(unique)} ALOS product(s) in total\n")
    return unique


# ================================================== multipart zip reading
class ConcatFile(io.RawIOBase):
    """Read-only, seekable view of several files as one continuous file."""

    def __init__(self, paths):
        super().__init__()
        self.paths = list(paths)
        self.sizes = [os.path.getsize(p) for p in self.paths]
        self.starts, s = [], 0
        for n in self.sizes:
            self.starts.append(s)
            s += n
        self.total = s
        self.pos = 0
        self._fh = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.total + offset
        return self.pos

    def readinto(self, b):
        want, done = len(b), 0
        while done < want and self.pos < self.total:
            i = max(k for k in range(len(self.starts)) if self.starts[k] <= self.pos)
            fh = self._fh.get(i)
            if fh is None:
                fh = self._fh[i] = open(self.paths[i], 'rb')
            fh.seek(self.pos - self.starts[i])
            n = min(want - done, self.sizes[i] - (self.pos - self.starts[i]))
            chunk = fh.read(n)
            if not chunk:
                break
            b[done:done + len(chunk)] = chunk
            done += len(chunk)
            self.pos += len(chunk)
        return done

    def read_at(self, offset, n):
        self.seek(offset)
        buf = bytearray(n)
        got = self.readinto(buf)
        return bytes(buf[:got])

    def close(self):
        for fh in self._fh.values():
            fh.close()
        self._fh = {}
        super().close()


class SplitZip:
    """Minimal zip reader for spanned (.z01 ... .zip) and byte-split
    (.zip.001 ...) archives, which Python's zipfile cannot open. Reads the
    central directory, then individual members, without extracting."""

    def __init__(self, paths):
        self.f = ConcatFile(paths)
        self.members = {}          # name -> (disk, local offset, csize, usize, method, flags)
        self.crcs = {}             # name -> CRC-32 from the central directory
        self._read_central_directory(len(paths))

    def _base(self, disk):
        if self.single_disk:
            return 0
        if disk >= len(self.f.starts):
            raise RuntimeError(f"part {disk + 1} of the multipart zip is missing")
        return self.f.starts[disk]

    def _read_central_directory(self, n_parts):
        tail_len = min(self.f.total, 65557 + 20)
        tail = self.f.read_at(self.f.total - tail_len, tail_len)
        i = tail.rfind(b'PK\x05\x06')
        if i < 0:
            raise RuntimeError("no zip end-of-central-directory record found "
                               "(last part missing or not a zip?)")
        eocd_pos = self.f.total - tail_len + i
        (_, disk_no, cd_disk, _, n_total, cd_size, cd_off,
         _) = struct.unpack('<4s4H2LH', tail[i:i + 22])
        self.single_disk = (disk_no == 0)        # byte-split: one logical disk
        if not self.single_disk and disk_no + 1 != n_parts:
            raise RuntimeError(f"spanned zip expects {disk_no + 1} parts, found {n_parts}")

        if 0xFFFF in (disk_no, cd_disk, n_total) or 0xFFFFFFFF in (cd_size, cd_off):
            loc = self.f.read_at(eocd_pos - 20, 20)                # zip64
            if loc[:4] != b'PK\x06\x07':
                raise RuntimeError("zip64 locator not found")
            _, z64_disk, z64_off, _ = struct.unpack('<4sLQL', loc)
            if disk_no == 0xFFFF:
                self.single_disk = False
            rec = self.f.read_at(self._base(z64_disk) + z64_off, 56)
            if rec[:4] != b'PK\x06\x06':
                raise RuntimeError("zip64 end-of-central-directory not found")
            (_, _, _, _, disk_no, cd_disk, _, n_total, cd_size,
             cd_off) = struct.unpack('<4sQ2H2L4Q', rec)
            self.single_disk = (disk_no == 0)

        cd = self.f.read_at(self._base(cd_disk) + cd_off, cd_size)
        p = 0
        while p + 46 <= len(cd) and cd[p:p + 4] == b'PK\x01\x02':
            (_, _, _, flags, method, _, _, crc, csize, usize, fnlen, exlen, cmlen,
             disk, _, _, loff) = struct.unpack('<4s6H3L5H2L', cd[p:p + 46])
            raw = cd[p + 46:p + 46 + fnlen]
            name = raw.decode('utf-8' if flags & 0x800 else 'cp437', 'replace')
            extra = cd[p + 46 + fnlen:p + 46 + fnlen + exlen]
            q = 0
            while q + 4 <= len(extra):                    # zip64 extra field
                hid, hlen = struct.unpack('<2H', extra[q:q + 4])
                if hid == 0x0001:
                    d = extra[q + 4:q + 4 + hlen]
                    k = 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack('<Q', d[k:k + 8])[0]; k += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack('<Q', d[k:k + 8])[0]; k += 8
                    if loff == 0xFFFFFFFF:
                        loff = struct.unpack('<Q', d[k:k + 8])[0]; k += 8
                    if disk == 0xFFFF:
                        disk = struct.unpack('<L', d[k:k + 4])[0]
                q += 4 + hlen
            self.members[name] = (disk, loff, csize, usize, method, flags)
            self.crcs[name] = crc
            p += 46 + fnlen + exlen + cmlen
            n_read = getattr(self, '_n_read', 0) + 1
            self._n_read = n_read
        if getattr(self, '_n_read', 0) != n_total:
            raise RuntimeError(f"zip directory not where the end record says: "
                               f"{getattr(self, '_n_read', 0)} of {n_total} entries found")

    def infolist(self):
        return [(n, v[3]) for n, v in self.members.items()]

    def _data_pos(self, name):
        """Offset of a member's (compressed) data in the joined parts."""
        disk, loff, csize, usize, method, flags = self.members[name]
        if flags & 1:
            raise RuntimeError(f"{name} is encrypted")
        pos = self._base(disk) + loff
        hdr = self.f.read_at(pos, 30)
        if hdr[:4] != b'PK\x03\x04':
            raise RuntimeError(f"bad local header for {name}")
        fnlen, exlen = struct.unpack('<2H', hdr[26:30])
        return pos + 30 + fnlen + exlen

    def read(self, name, limit=None):
        disk, loff, csize, usize, method, flags = self.members[name]
        data_pos = self._data_pos(name)
        limit = usize if limit is None else min(limit, usize)
        if method == 0:
            return self.f.read_at(data_pos, limit)
        if method != 8:
            raise RuntimeError(f"{name}: compression method {method} not supported")
        dec, out, done = zlib.decompressobj(-15), bytearray(), 0
        self.f.seek(data_pos)
        while done < csize and len(out) < limit:
            chunk = self.f.read(min(65536, csize - done))
            if not chunk:
                break
            done += len(chunk)
            out += dec.decompress(chunk)
        return bytes(out[:limit])

    def read_ranges(self, name, ranges):
        """[(offset, n), ...] (ascending) of the uncompressed member, read in
        one forward pass. Stored members are read directly; deflated members
        are decompressed up to the last range (nothing is written to disk)."""
        disk, loff, csize, usize, method, flags = self.members[name]
        data_pos = self._data_pos(name)
        if method == 0:
            return [self.f.read_at(data_pos + o, n) for o, n in ranges]
        if method != 8:
            raise RuntimeError(f"{name}: compression method {method} not supported")
        out, dec, done = [], zlib.decompressobj(-15), 0
        upos = 0                         # uncompressed bytes produced so far
        buf, buf_start = bytearray(), 0  # buf holds bytes [buf_start, upos)
        self.f.seek(data_pos)
        for off, n in ranges:
            while upos < off + n and done < csize:
                chunk = self.f.read(min(1 << 20, csize - done))
                if not chunk:
                    break
                done += len(chunk)
                data = dec.decompress(chunk)
                new_upos = upos + len(data)
                if new_upos <= off:                      # all before the range
                    buf, buf_start = bytearray(), new_upos
                elif upos < off:                         # range starts inside
                    buf, buf_start = bytearray(data[off - upos:]), off
                else:
                    buf += data
                upos = new_upos
            out.append(bytes(buf[off - buf_start:off - buf_start + n])
                       if buf_start <= off else b'')
            if buf_start < off:                          # drop what is behind
                del buf[:off - buf_start]
                buf_start = off
        return out

    def iter_member(self, name, chunk=1 << 22):
        """Uncompressed data of a member, in chunks (for --verify)."""
        disk, loff, csize, usize, method, flags = self.members[name]
        pos, done = self._data_pos(name), 0
        dec = zlib.decompressobj(-15) if method == 8 else None
        if method not in (0, 8):
            raise RuntimeError(f"{name}: compression method {method} not supported")
        self.f.seek(pos)
        while done < csize:
            data = self.f.read(min(chunk, csize - done))
            if not data:
                raise RuntimeError(f"{name}: data ends {csize - done:,} bytes early")
            done += len(data)
            yield dec.decompress(data) if dec else data
        if dec:
            yield dec.flush()

    def close(self):
        self.f.close()


class RecoveredZip(SplitZip):
    """Reader for a damaged archive, built from a forward scan of its local
    file headers (see scan_local_headers) instead of its central directory.
    Only members whose data is completely present are included."""

    def __init__(self, paths, members):
        self.f = ConcatFile(paths)
        self.single_disk = True
        self.members, self._pos, self.crcs = {}, {}, {}
        for name, data_pos, csize, usize, method, flags in members:
            self.members[name] = (0, 0, csize, usize, method, flags)
            self._pos[name] = data_pos

    def _data_pos(self, name):
        if self.members[name][5] & 1:
            raise RuntimeError(f"{name} is encrypted")
        return self._pos[name]


class ArchiveError(Exception):
    """A product whose zip(s) cannot be read at all; .diagnosis says why."""

    def __init__(self, message, diagnosis=''):
        super().__init__(message)
        self.diagnosis = diagnosis


ZIP_SIGNATURES = {b'PK\x03\x04': 'a zip file header',
                  b'PK\x07\x08': 'a split-archive marker',
                  b'PK\x05\x06': 'a zip end record (empty zip)',
                  b'PK\x01\x02': 'a zip directory entry',
                  b'PK\x06\x06': 'a zip64 end record'}


def find_end_record(f, start, end):
    """Last zip end-of-central-directory record in [start, end) of a
    ConcatFile (the record is in the last 22 + 65535 bytes). (pos, bytes)."""
    n = min(end - start, 65557 + 20)
    if n < 22:
        return None, None
    tail = f.read_at(end - n, n)
    i = tail.rfind(b'PK\x05\x06')
    while i >= 0 and len(tail) - i < 22:
        i = tail.rfind(b'PK\x05\x06', 0, i)
    return (end - n + i, tail[i:i + 22]) if i >= 0 else (None, None)


def scan_local_headers(f, max_members=100000):
    """Walk the zip's local file headers from the start of the joined parts,
    as a zip repair tool would. Returns (members, stop_reason); members are
    (name, data_pos, csize, usize, method, flags) for files whose data is
    completely present."""
    pos, out = 0, []
    if f.read_at(0, 4) == b'PK\x07\x08':          # split-archive marker
        pos = 4
    while len(out) < max_members:
        hdr = f.read_at(pos, 30)
        if not hdr:
            return out, f"end of the data after {len(out)} file(s), with no zip directory after them"
        if hdr[:4] in (b'PK\x01\x02', b'PK\x05\x06', b'PK\x06\x06'):
            return out, f"reached the zip directory at offset {pos:,}: all file data is present"
        if len(hdr) >= 4 and hdr[:4] != b'PK\x03\x04':
            return out, (f"unexpected bytes {hdr[:4].hex()} at offset {pos:,} where a file header "
                         f"should start")
        if len(hdr) < 30:
            return out, f"data ends inside a file header at offset {pos:,}"
        (_, _, flags, method, _, _, _, csize, usize, fnlen,
         exlen) = struct.unpack('<4s5H3L2H', hdr)
        ne = f.read_at(pos + 30, fnlen + exlen)
        if len(ne) < fnlen + exlen:
            return out, f"data ends inside a file header at offset {pos:,}"
        name = ne[:fnlen].decode('utf-8' if flags & 0x800 else 'cp437', 'replace')
        extra, zip64, q = ne[fnlen:], False, 0
        while q + 4 <= len(extra):
            hid, hlen = struct.unpack('<2H', extra[q:q + 4])
            if hid == 0x0001 and (usize == 0xFFFFFFFF or csize == 0xFFFFFFFF):
                d, zip64 = extra[q + 4:q + 4 + hlen], True
                if len(d) >= 8:
                    usize = struct.unpack('<Q', d[0:8])[0]
                if len(d) >= 16:
                    csize = struct.unpack('<Q', d[8:16])[0]
            q += 4 + hlen
        data_pos = pos + 30 + fnlen + exlen
        if flags & 0x08 and csize == 0:
            return out, (f"{name}: its size is only stored after its data, so the scan "
                         f"cannot skip over it")
        end = data_pos + csize
        if end > f.total:
            return out, (f"TRUNCATED inside {name}: it needs {csize:,} bytes but the data "
                         f"ends {end - f.total:,} bytes short")
        out.append((name, data_pos, csize, usize, method, flags))
        pos = end
        if flags & 0x08:                                    # data descriptor
            if f.read_at(pos, 4) == b'PK\x07\x08':
                pos += 4
            pos += 4 + (16 if zip64 else 8)
    return out, "stopped after too many files"


def diagnose(paths):
    """Structural check of a zip or multipart set, without extracting.
    Returns (lines, members_found_by_scan, scan_stop_reason)."""
    f = ConcatFile(paths)
    lines = []
    try:
        for i, p in enumerate(paths):
            start, size = f.starts[i], f.sizes[i]
            head = f.read_at(start, 4)
            what = (ZIP_SIGNATURES.get(head, f"bytes {head.hex()} (not a zip header)")
                    if size else 'nothing (empty file)')
            pos, _ = find_end_record(f, start, start + size)
            lines.append(f"part {i + 1}/{len(paths)} {os.path.basename(p)}: {size:,} bytes, "
                         f"starts with {what}, "
                         + ("ends with a zip end record" if pos is not None else "no zip end record at its end"))

        pos, rec = find_end_record(f, 0, f.total)
        if pos is None:
            lines.append("no zip end record at the end of the last part, so the archive's "
                         "directory is missing")
        else:
            (_, disk_no, cd_disk, _, n_total, cd_size, cd_off,
             _) = struct.unpack('<4s4H2LH', rec)
            lines.append(f"end record at offset {pos:,}: disk {disk_no}, directory on disk {cd_disk}, "
                         f"{n_total} entries, directory of {cd_size:,} bytes at offset {cd_off:,}")
            rec_pos = pos
            if 0xFFFF in (disk_no, cd_disk, n_total) or 0xFFFFFFFF in (cd_size, cd_off):
                loc = f.read_at(pos - 20, 20)
                if loc[:4] != b'PK\x06\x07':
                    lines.append("zip64 locator missing just before the end record")
                else:
                    _, zdisk, zoff, ztotal = struct.unpack('<4sLQL', loc)
                    lines.append(f"zip64 locator: zip64 end record on disk {zdisk} at offset "
                                 f"{zoff:,}; archive has {ztotal} disk(s)/part(s)")
                    if ztotal not in (1, len(paths)):
                        lines.append(f"PART COUNT MISMATCH: the archive says {ztotal} part(s), "
                                     f"{len(paths)} found")
                    lo = max(0, pos - 20 - 65536)
                    near = f.read_at(lo, pos - 20 - lo)
                    j = near.rfind(b'PK\x06\x06')
                    if j < 0:
                        lines.append("zip64 end record not found before the locator")
                    else:
                        actual = lo + j
                        (_, _, _, _, zd, zcd, _, zn, zsize,
                         zcdoff) = struct.unpack('<4sQ2H2L4Q', f.read_at(actual, 56))
                        shift = actual - zoff
                        lines.append(f"zip64 end record found at offset {actual:,}"
                                     + (f" ({shift:+,} bytes from where the locator says)" if shift else "")
                                     + f": disk {zd}, {zn} entries, directory of {zsize:,} bytes "
                                       f"at offset {zcdoff:,}")
                        cd_size, cd_off, rec_pos = zsize, zcdoff, actual
            cd_actual = rec_pos - cd_size        # the directory ends where the end records start
            if f.read_at(cd_off, 4) == b'PK\x01\x02':
                lines.append("zip directory is where the end record says")
            elif cd_actual >= 0 and f.read_at(cd_actual, 4) == b'PK\x01\x02':
                lines.append(f"zip directory found at offset {cd_actual:,}, {cd_actual - cd_off:+,} "
                             f"bytes from where the end record says (offsets counted from a "
                             f"different start, e.g. per part)")
            else:
                lines.append("zip directory not found where expected")

        members, reason = scan_local_headers(f)
        lines.append(f"scan of file headers: {len(members)} complete file(s); {reason}")
        if members:
            lines.append(f"last complete file: {members[-1][0]}")
        if reason.startswith('TRUNCATED') or reason.startswith('data ends'):
            lines.append("CONCLUSION: INCOMPLETE - the data stops early. A part is missing or "
                         "cut short (interrupted download or copy), or the parts are out of "
                         "order. Compare part sizes with another copy.")
        elif reason.startswith('end of the data'):
            lines.append("CONCLUSION: INCOMPLETE - all file data is present but the zip "
                         "directory is missing: the last part is probably missing or cut short.")
        elif reason.startswith('reached the zip directory'):
            lines.append("CONCLUSION: file data complete; only the zip's end/directory records "
                         "are inconsistent, so the files can still be read.")
        elif reason.startswith('unexpected bytes'):
            lines.append("CONCLUSION: DAMAGED or not a zip, or the parts are out of order.")
        return lines, members, reason
    finally:
        f.close()


class Archive:
    """One product: a single zip or a multipart set, read without extracting.
    If the zip structure is broken, falls back to a forward scan of the file
    headers and reads whatever is intact (self.damage / self.diagnosis)."""

    def __init__(self, entry):
        self.paths = [entry] if isinstance(entry, str) else list(entry)
        self.damage, self.diagnosis = '', ''
        self.readers, self.kind = [], 'zip'
        try:
            self._open()
        except Exception as e:
            for r in self.readers:
                try:
                    r.close()
                except Exception:
                    pass
            self.readers = []
            lines, members, reason = diagnose(self.paths)
            if not members:
                raise ArchiveError(f"{type(e).__name__}: {e}", ' | '.join(lines))
            self.damage = f"{type(e).__name__}: {e}"
            self.diagnosis = ' | '.join(lines)
            self.kind = f"{self.kind} (damaged, {len(members)} file(s) recovered)"
            self.readers.append(RecoveredZip(self.paths, members))
        self.index = {}                # member name -> (reader, size)
        for r in self.readers:
            if isinstance(r, zipfile.ZipFile):
                items = [(i.filename, i.file_size) for i in r.infolist()]
            else:
                items = r.infolist()
            for name, size in items:
                self.index.setdefault(name, (r, size))

    def _open(self):
        names = [os.path.basename(p).lower() for p in self.paths]
        if len(self.paths) == 1 and names[0].endswith('.zip'):
            self.kind = 'zip'
            self.readers.append(zipfile.ZipFile(self.paths[0]))
        elif any(re.search(r'\.z\d{2,}$', n) for n in names):
            self.kind = 'spanned'
            if not names[-1].endswith('.zip'):
                raise RuntimeError("spanned zip is missing its final .zip part")
            self.readers.append(SplitZip(self.paths))
        elif any(re.search(r'\.zip\.\d{3,}$', n) for n in names):
            # .zip.001 / .zip.0001 parts: usually one zip cut into chunks, but
            # if every part is a complete zip on its own, read them separately
            if len(self.paths) > 1 and all(zipfile.is_zipfile(p) for p in self.paths):
                self.kind = 'separate-parts'
                for p in self.paths:
                    self.readers.append(zipfile.ZipFile(p))
            else:
                self.kind = 'byte-split'
                self.readers.append(SplitZip(self.paths))
        else:
            self.kind = 'separate-parts'
            for p in self.paths:
                self.readers.append(zipfile.ZipFile(p))

    def members(self):
        return [(n, v[1]) for n, v in self.index.items()]

    def read(self, name, limit=None):
        r, _ = self.index[name]
        if isinstance(r, zipfile.ZipFile):
            with r.open(name) as fh:
                return fh.read() if limit is None else fh.read(limit)
        return r.read(name, limit)

    def read_ranges(self, name, ranges):
        """Read several (offset, n) ranges of one member in one forward pass."""
        r, _ = self.index[name]
        if isinstance(r, zipfile.ZipFile):
            out = []
            with r.open(name) as fh:
                for off, n in ranges:
                    fh.seek(off)
                    out.append(fh.read(n))
            return out
        return r.read_ranges(name, ranges)

    def gdal_path(self, name):
        """/vsizip/ path for GDAL (single zips only)."""
        return f"/vsizip/{self.paths[0]}/{name}" if self.kind == 'zip' else None

    def close(self):
        for r in self.readers:
            r.close()


# ============================================== stage 1b: metadata parsing
NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
VECTOR_RE = re.compile(rf'^\s*{NUM}(?:\s*[,;\s]\s*{NUM})+\s*$')


def is_vector(value):
    """True for '1.0 2.0 3.0' or '1,2,3' style values (vectors / tables)."""
    return bool(VECTOR_RE.match(value))


def clean(value):
    return re.sub(r'[\r\n\t]+', ' ', value).strip()


def parse_kv_text(text):
    """key="value" lines (JAXA summary.txt / workreport)."""
    out = {}
    for line in text.splitlines():
        m = re.match(r'^\s*([A-Za-z_][\w.\-]*)\s*=\s*"?(.*?)"?\s*$', line)
        if m:
            key, val = m.group(1), clean(m.group(2))
            if val and not is_vector(val) and key not in out:
                out[key] = val
    return out


def parse_xml(data):
    """Leaf element values that occur exactly once (repeated = vector)."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return {}
    counts, values = {}, {}

    def walk(el, path):
        tag = el.tag.split('}')[-1]
        path = path + (tag,)
        kids = list(el)
        if not kids:
            key = '.'.join(path[-2:])
            counts[key] = counts.get(key, 0) + 1
            if el.text and el.text.strip():
                values[key] = clean(el.text)
        for k in kids:
            walk(k, path)

    walk(root, ())
    return {k: v for k, v in values.items()
            if counts[k] == 1 and not is_vector(v)}


def parse_led(b):
    """CEOS leader file: scalar fields of the data set summary record.
    Byte positions (1-based) are from JAXA's PALSAR-2 CEOS format description
    (4096-byte record). ALOS-4 PALSAR-3 uses a 4680-byte record whose first
    1462 bytes are the same and whose later fields are shifted (orbit
    direction at 1823, off-nadir at 2435). Each value is sanity-checked."""
    out = {}
    if len(b) < 720 + 1854:
        return out
    fdr_len = struct.unpack('>I', b[8:12])[0]
    start = fdr_len if 100 <= fdr_len <= 10000 else 720
    dss_len = struct.unpack('>I', b[start + 8:start + 12])[0]
    dss = b[start:start + max(4096, min(dss_len, 8192))]
    if len(dss) < 1854:
        return out
    palsar3 = (dss_len == 4680)
    out['LED_dss_record_length'] = str(dss_len)
    pos_time_dir = (1823, 1830) if palsar3 else (1535, 1542)
    pos_off_nadir = (2435, 2450) if palsar3 else (1839, 1854)

    def txt(a, z):
        return dss[a - 1:z].decode('ascii', 'replace').strip()

    def num(a, z, lo, hi):
        try:
            v = float(txt(a, z))
        except ValueError:
            return None
        return v if lo <= v <= hi else None

    s = txt(69, 100)
    if re.match(r'^\d{14}', s):
        out['LED_scene_centre_time'] = s
    for key, (a, z, lo, hi) in {
            'LED_scene_centre_lat': (117, 132, -90, 90),
            'LED_scene_centre_lon': (133, 148, -180, 360),
            'LED_incidence_angle_centre': (485, 492, 0.01, 89.99),
            'LED_radar_wavelength_m': (501, 516, 0.001, 1.0),
            'LED_nominal_off_nadir_angle': pos_off_nadir + (0.01, 89.99)}.items():
        v = num(a, z, lo, hi)
        if v is not None:
            out[key] = f"{v:g}"
    for key, (a, z) in {'LED_platform_id': (397, 412),
                        'LED_sensor_id_mode': (413, 444)}.items():
        v = txt(a, z)
        if v and v.isprintable():
            out[key] = v
    v = txt(*pos_time_dir)
    if v.upper().startswith(('ASC', 'DESC')):
        out['LED_time_direction'] = v
    o = txt(445, 452)
    if o.isdigit():
        out['LED_orbit_number'] = str(int(o))
    return out


def ceos_image_footprint(archive, members):
    """Corners and centre from CEOS image files (IMG-XX-..., no extension).
    Every image line record holds the lat/lon of its first, middle and last
    pixel as big-endian signed 4-byte integers in 1/1,000,000 degree:
    bytes 193-216 in Level 1.1 signal data records, 133-156 in Level 1.5
    processed data records (JAXA PALSAR CEOS format description). The first
    and last records give 4 corners, the middle record the scene centre.
    ScanSAR products have one image file per scan; their corners are
    combined. Returns (points [(lat, lon) ...], centre (lat, lon)) or (None, None)."""
    imgs = [(n, s) for n, s in members
            if re.match(r'^img-', os.path.basename(n).lower())
            and not re.search(r'\.(tif|tiff|jpg|png|xml|txt)$', n.lower())]
    if not imgs:
        return None, None

    def pol(n):
        m = re.match(r'^img-([a-z0-9]{2})-', os.path.basename(n).lower())
        return m.group(1) if m else ''

    first_pol = sorted({pol(n) for n, _ in imgs})[0]
    chosen = sorted([(n, s) for n, s in imgs if pol(n) == first_pol])[:10]

    def latlon(rec, start):
        if len(rec) < start + 23:
            return None
        v = struct.unpack('>6i', rec[start - 1:start + 23])
        lats, lons = [x / 1e6 for x in v[:3]], [x / 1e6 for x in v[3:]]
        if any(abs(a) > 90 for a in lats) or any(abs(o) > 180 for o in lons):
            return None
        if all(x == 0 for x in v):
            return None
        f, m, l = list(zip(lats, lons))     # first, middle, last pixel
        width = haversine_km(f[0], f[1], l[0], l[1])
        mid_err = haversine_km(m[0], m[1], (f[0] + l[0]) / 2, (f[1] + l[1]) / 2)
        if not (1 < width < 1000) or mid_err > 0.25 * width:
            return None                     # not a plausible line of a scene
        return [f, m, l]

    points, centres = [], []
    for name, size in chosen:
        head = archive.read_ranges(name, [(0, 720 + 260)])[0]
        if len(head) < 720 + 12:
            continue
        reclen = struct.unpack('>I', head[720 + 8:720 + 12])[0]
        if reclen < 260 or size < 720 + reclen:
            continue
        nrec = (size - 720) // reclen
        first, mid, last = archive.read_ranges(
            name, [(720, 260), (720 + (nrec // 2) * reclen, 260),
                   (720 + (nrec - 1) * reclen, 260)])
        for start in (193, 133):            # Level 1.1, then Level 1.5 layout
            a, m, z = latlon(first, start), latlon(mid, start), latlon(last, start)
            if a and z and max_span_km([a[0], a[2], z[0], z[2]]) < 2000:
                points += [a[0], a[2], z[0], z[2]]
                if m:
                    centres.append(m[1])
                break
    if not points:
        return None, None
    centre = centres[len(centres) // 2] if centres else None
    return points, centre


def geotiff_footprint(archive, members):
    """Corners and centre of a GeoTIFF image inside the zip, via GDAL's
    /vsizip/ (single zips only; needs the osgeo bindings)."""
    tifs = [n for n, s in members if re.search(r'\.tiff?$', n.lower())]
    if not tifs or archive.gdal_path(tifs[0]) is None:
        return None, None
    try:
        from osgeo import gdal, osr
        gdal.UseExceptions()
        gdal.PushErrorHandler('CPLQuietErrorHandler')   # e.g. harmless TIFF tag warnings
        try:
            ds = gdal.Open(archive.gdal_path(tifs[0]))
        finally:
            gdal.PopErrorHandler()
        gt, w, h = ds.GetGeoTransform(), ds.RasterXSize, ds.RasterYSize
        src = osr.SpatialReference()
        src.ImportFromWkt(ds.GetProjection())
        dst = osr.SpatialReference()
        dst.ImportFromEPSG(4326)
        for s in (src, dst):
            s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        tr = osr.CoordinateTransformation(src, dst)
        pts = []
        for px, py in ((0, 0), (w, 0), (0, h), (w, h), (w / 2, h / 2)):
            x = gt[0] + px * gt[1] + py * gt[2]
            y = gt[3] + px * gt[4] + py * gt[5]
            lon, lat = tr.TransformPoint(x, y)[:2]
            pts.append((lat, lon))
        return pts[:4], pts[4]
    except Exception:
        return None, None


def max_span_km(pts):
    return max(haversine_km(a[0], a[1], b[0], b[1]) for a in pts for b in pts)


def summary_corners(raw):
    """4 corners from summary keys: Img_ImageScene*/Img_FrameScene* first,
    then any key naming a corner (LeftTop, UpperLeft, ...) with Lat/Lon."""
    for src in ('ImageScene', 'FrameScene'):
        cand = [(to_float(raw.get(f'Img_{src}{c}Latitude')), to_float(raw.get(f'Img_{src}{c}Longitude')))
                for c in ('LeftTop', 'RightTop', 'LeftBottom', 'RightBottom')]
        if all(a is not None and b is not None for a, b in cand):
            return cand, f'summary {src}'
    names = {'lefttop': 'lt', 'topleft': 'lt', 'upperleft': 'lt',
             'righttop': 'rt', 'topright': 'rt', 'upperright': 'rt',
             'leftbottom': 'lb', 'bottomleft': 'lb', 'lowerleft': 'lb',
             'rightbottom': 'rb', 'bottomright': 'rb', 'lowerright': 'rb'}
    found = {}
    for key in sorted(raw, key=lambda k: ('image' not in k.lower(), k)):
        flat = re.sub(r'[^a-z]', '', key.lower())
        corner = next((v for k, v in names.items() if k in flat), None)
        coord = 'lat' if 'lat' in flat else 'lon' if 'lon' in flat else None
        val = to_float(raw[key])
        if corner and coord and val is not None:
            found.setdefault((corner, coord), val)
    if len(found) == 8:
        return [(found[(c, 'lat')], found[(c, 'lon')]) for c in ('lt', 'rt', 'lb', 'rb')], 'summary corner keys'
    return None, ''


def parse_kml(data):
    """Footprint from a JAXA product KML. Returns ([(lat, lon) x 4], kind)
    or (None, ''). Tries gx:LatLonQuad, then a LinearRing, then LatLonBox."""
    t = data.decode('utf-8', 'replace')

    def coords(s):
        pts = []
        for tok in s.split():
            p = tok.split(',')
            if len(p) >= 2:
                lon, lat = to_float(p[0]), to_float(p[1])
                if lon is not None and lat is not None and -90 <= lat <= 90:
                    pts.append((lat, lon))
        return pts

    m = re.search(r'<(?:\w+:)?LatLonQuad\b[^>]*>.*?<(?:\w+:)?coordinates>(.*?)</', t, re.S)
    if m:
        pts = coords(m.group(1))
        if len(pts) >= 4:
            return pts[:4], 'KML LatLonQuad'
    m = re.search(r'<(?:\w+:)?LinearRing\b[^>]*>.*?<(?:\w+:)?coordinates>(.*?)</', t, re.S)
    if m:
        pts = coords(m.group(1))
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]                       # drop the closing point
        if len(pts) == 4:
            return pts, 'KML polygon'
        if len(pts) > 4:                         # densified ring: extreme points
            return [max(pts, key=lambda p: p[0] - p[1]),     # NW
                    max(pts, key=lambda p: p[0] + p[1]),     # NE
                    min(pts, key=lambda p: p[0] + p[1]),     # SW
                    min(pts, key=lambda p: p[0] - p[1])], 'KML polygon'
    box = {}
    for side in ('north', 'south', 'east', 'west'):
        mm = re.search(rf'<(?:\w+:)?{side}>\s*([-+0-9.eE]+)\s*</', t)
        if mm:
            box[side] = to_float(mm.group(1))
    if len(box) == 4 and None not in box.values():
        n, s, e, w = box['north'], box['south'], box['east'], box['west']
        return [(n, w), (n, e), (s, w), (s, e)], 'KML LatLonBox'
    return None, ''


def geo_corners(pts):
    """(lat, lon) points -> {'nw': .., 'ne': .., 'sw': .., 'se': ..}.
    4 points: the two most northerly are NW/NE, the smaller longitude of a
    pair is west. More points (ScanSAR scans, densified outlines): the
    extreme points in the NW/NE/SW/SE directions."""
    if len(pts) != 4:
        return {'nw': max(pts, key=lambda p: p[0] - p[1]),
                'ne': max(pts, key=lambda p: p[0] + p[1]),
                'sw': min(pts, key=lambda p: p[0] + p[1]),
                'se': min(pts, key=lambda p: p[0] - p[1])}
    by_lat = sorted(pts, key=lambda p: -p[0])
    nw, ne = sorted(by_lat[:2], key=lambda p: p[1])
    sw, se = sorted(by_lat[2:], key=lambda p: p[1])
    return {'nw': nw, 'ne': ne, 'sw': sw, 'se': se}


def iso_time(s):
    """'20150101 12:00:00.123' / '20150101120000123' -> ISO 8601 text."""
    if not s:
        return ''
    m = re.match(r'^(\d{4})(\d{2})(\d{2})[ T]?(\d{2}):?(\d{2}):?(\d{2})(?:\.?(\d+))?', s.strip())
    if not m:
        return s
    frac = (m.group(7) or '')[:6]
    t = f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:{m.group(6)}"
    return t + ('.' + frac if frac else '')


def to_float(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def standard_fields(raw, member_names, entry_paths, footprint=None):
    """The standardised (first) CSV columns, from raw values and file names.
    footprint: (points, corner source, centre (lat, lon), centre source)
    found by extract_metadata from the summary, KML or image files."""
    std = {}
    get = raw.get
    img = [os.path.basename(n) for n in member_names
           if os.path.basename(n).upper().startswith('IMG-')]

    pols = sorted({m.group(1) for n in img
                   for m in [re.match(r'IMG-([HV]{2})\b', n, re.I)] if m},
                  key=lambda p: ['HH', 'HV', 'VH', 'VV'].index(p.upper())
                  if p.upper() in ('HH', 'HV', 'VH', 'VV') else 9)
    std['polarizations'] = '+'.join(p.upper() for p in pols)

    scene_id, product_id = get('Scs_SceneID', ''), get('Pds_ProductID', '')
    for n in img:
        m = re.search(r'IMG-[HV]{2}-(?:[A-Z0-9]+-)?([A-Z0-9]+\d-\d{6})-([A-Z0-9_.]{10})', n, re.I)
        if m:
            scene_id = scene_id or m.group(1)
            product_id = product_id or m.group(2)
            break
    std['scene_id'], std['product_id'] = scene_id, product_id

    # ALOS-2 product ID DDDEFFFGHI, e.g. FBDR1.1__A (other formats left alone)
    if re.match(r'^[A-Z]{3}[LR]\d\.\d[A-Z_][A-Z_][AD]$', product_id, re.I):
        std['observation_mode'] = product_id[0:3]
        std['look_direction'] = {'L': 'left', 'R': 'right'}.get(product_id[3].upper(), '')
        std['processing_level'] = product_id[4:7]
        std['orbit_direction'] = {'A': 'ascending', 'D': 'descending'}.get(product_id[9].upper(), '')
    else:
        # ALOS-4 code after the date in the scene ID / file name, e.g.
        # ALOS41712610250427HBQ-RD0606: mode HBQ, R(ight)-looking, D(escending)
        for text in (scene_id, os.path.basename(entry_paths[0])):
            m4 = re.search(r'\d{6}([A-Z]{3})[-A-Z]([LR])([AD])\d{4}', text or '')
            if m4:
                std['observation_mode'] = m4.group(1)
                std['look_direction'] = {'L': 'left', 'R': 'right'}[m4.group(2)]
                std['orbit_direction'] = {'A': 'ascending', 'D': 'descending'}[m4.group(3)]
                break
    if get('Lbi_ProcessLevel'):
        std['processing_level'] = get('Lbi_ProcessLevel')
    if not std.get('orbit_direction') and get('LED_time_direction'):
        std['orbit_direction'] = ('ascending' if get('LED_time_direction').upper().startswith('ASC')
                                  else 'descending')

    m = re.match(r'^([A-Z0-9]{5})(\d{5})(\d{4})-(\d{6})$', scene_id, re.I)
    if m:
        std['orbit_number'] = str(int(m.group(2)))
        std['frame_number'] = m.group(3)
    std['orbit_number'] = std.get('orbit_number') or get('LED_orbit_number', '')

    sat = get('Lbi_Satellite') or (m.group(1) if m else '') or get('LED_platform_id', '')
    if not sat:
        fm = re.search(r'ALOS-?\d', os.path.basename(entry_paths[0]), re.I)
        sat = fm.group(0).upper().replace('-', '') if fm else ''
    std['satellite'] = sat

    std['acq_start'] = iso_time(get('Img_SceneStartDateTime', ''))
    std['acq_center'] = iso_time(get('Img_SceneCenterDateTime', '') or get('LED_scene_centre_time', ''))
    std['acq_end'] = iso_time(get('Img_SceneEndDateTime', ''))

    inc = get('LED_incidence_angle_centre')
    if not inc:
        cands = [k for k in raw if 'incidence' in k.lower() and to_float(raw[k]) is not None]
        cands.sort(key=lambda k: ('cent' not in k.lower(), k))
        inc = raw[cands[0]] if cands else ''
    std['incidence_angle_center'] = inc
    std['off_nadir_angle'] = get('Img_OffNadirAngle') or get('LED_nominal_off_nadir_angle', '')
    std['pixel_spacing'] = get('Pds_PixelSpacing', '')

    pts, source, centre, centre_src = footprint or (None, '', None, '')
    if pts:
        for name, (lat, lon) in geo_corners(pts).items():
            std[name + '_lat'], std[name + '_lon'] = f"{lat:.7f}", f"{lon:.7f}"
        std['corner_source'] = source

    # scene centre: summary keys, leader file (Level 1.5+), image files,
    # then the mean of the corners
    for src, lat_key, lon_key in (
            ('summary', 'Img_ImageSceneCenterLatitude', 'Img_ImageSceneCenterLongitude'),
            ('summary', 'Img_FrameSceneCenterLatitude', 'Img_FrameSceneCenterLongitude'),
            ('leader file', 'LED_scene_centre_lat', 'LED_scene_centre_lon')):
        lat, lon = to_float(get(lat_key)), to_float(get(lon_key))
        if lat is not None and lon is not None:
            centre, centre_src = (lat, lon), src
            break
    if centre is None:
        keys = sorted(k for k in raw if re.search(r'cent(er|re)', k, re.I))
        lat = next((to_float(raw[k]) for k in keys if re.search(r'lat', k, re.I)
                    and to_float(raw[k]) is not None), None)
        lon = next((to_float(raw[k]) for k in keys if re.search(r'lon', k, re.I)
                    and to_float(raw[k]) is not None), None)
        if lat is not None and lon is not None:
            centre, centre_src = (lat, lon), 'summary centre keys'
    if centre is None and pts:
        centre = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        centre_src = 'mean of corners'
    if centre is not None:
        std['center_lat'], std['center_lon'] = f"{centre[0]:.7f}", f"{centre[1]:.7f}"
        std['center_source'] = centre_src
    return std


def product_name(entry_paths):
    """Base name of the product, without extension or part numbering."""
    n = os.path.basename(entry_paths[0])
    n = re.sub(r'\.zip\.\d{3,}$', '', n, flags=re.I)
    n = re.sub(r'\.z\d{2,}$', '', n, flags=re.I)
    m = PART_REGEX.match(n)
    if m:
        return m.group('base')
    return re.sub(r'\.zip$', '', n, flags=re.I)


def extract_metadata(job):
    """Worker: all scalar metadata of one entry (single zip or multipart)."""
    idx, entry = job
    paths = [entry] if isinstance(entry, str) else list(entry)
    row = {'product': product_name(paths), 'n_parts': str(len(paths)),
           'zip_path': paths[0], 'all_parts': ';'.join(paths)}
    raw, meta_files = {}, []
    archive = None
    try:
        archive = Archive(entry)
        row['archive_type'] = archive.kind
        members = archive.members()
        names = [n for n, _ in members]

        def base(n):
            return os.path.basename(n).lower()

        # ALOS-2: summary.txt; ALOS-4: summary-<scene ID>.txt
        order = ([n for n, s in members if re.search(r'summary.*\.txt$', base(n))] +
                 [n for n, s in members if base(n).startswith('workreport')])
        for n in order:
            if dict(members)[n] <= MAX_TEXT_MEMBER:
                text = archive.read(n).decode('utf-8', 'replace')
                for k, v in parse_kv_text(text).items():
                    raw.setdefault(k, v)
                meta_files.append(n)

        for n in [n for n, s in members if base(n).startswith('led-')][:1]:
            for k, v in parse_led(archive.read(n, LED_BYTES)).items():
                raw.setdefault(k, v)
            meta_files.append(n)

        for n, s in members:
            if base(n).endswith('.xml') and s <= MAX_TEXT_MEMBER:
                for k, v in parse_xml(archive.read(n)).items():
                    raw.setdefault('XML_' + k, v)
                meta_files.append(n)

        # corners, cheapest source first; image files only when needed
        pts, source = summary_corners(raw)
        centre, centre_src = None, ''
        if not pts:
            for n, s in members:
                if base(n).endswith('.kml') and s <= MAX_TEXT_MEMBER:
                    pts, source = parse_kml(archive.read(n))
                    meta_files.append(n)
                    if pts:
                        break
        if not pts:
            img_pts, img_centre = ceos_image_footprint(archive, members)
            if img_pts:
                pts, source = img_pts, 'CEOS image records'
            if img_centre:
                centre, centre_src = img_centre, 'CEOS image records'
        if not pts:
            tif_pts, tif_centre = geotiff_footprint(archive, members)
            if tif_pts:
                pts, source = tif_pts, 'GeoTIFF'
                centre, centre_src = centre or tif_centre, centre_src or 'GeoTIFF'

        row.update(standard_fields(raw, names, paths, (pts, source, centre, centre_src)))
        files = [n for n in names if not n.endswith('/')]
        shown = ', '.join(sorted({base(n) for n in files})[:8]) + (' ...' if len(files) > 8 else '')
        jaxa_like = any(re.match(r'^(summary.*\.txt|workreport.*|led-.*|img-.*|vol-.*|trl-.*|brs-.*)$',
                                 base(n)) or base(n).endswith('.kml') for n in files)
        problems = []
        if archive.damage:
            status = 'damaged_recovered'
            problems.append(f"damaged zip ({archive.damage}); metadata read from the "
                            f"{len(files)} intact file(s)")
            row['diagnosis'] = archive.diagnosis
        if not jaxa_like and not pts:
            if not archive.damage:       # damaged: JAXA files may be in the lost part
                status = 'not_jaxa_product'
            problems.append(f"zip holds no JAXA product files ({len(files)} file(s): {shown})")
        elif 'nw_lat' not in row:
            status = status if archive.damage else 'no_footprint'
            problems.append('no corner coordinates found'
                            + ('; scene centre only' if 'center_lat' in row else '')
                            + f'; zip holds: {shown}')
        elif not archive.damage:
            status = 'ok'
        row['status'] = status
        if problems:
            row['error'] = '; '.join(problems)
    except ArchiveError as e:                   # unreadable: record why
        row['status'] = 'damaged_unreadable'
        row['error'] = str(e)
        row['diagnosis'] = e.diagnosis
    except Exception as e:                      # keep going; record the problem
        row['status'] = 'error'
        row['error'] = f"{type(e).__name__}: {e}"
    finally:
        if archive is not None:
            archive.close()
    row['meta_files'] = ';'.join(meta_files)
    for k, v in raw.items():
        if k not in row:
            row[k] = v
    return idx, row


def stage_metadata(entries):
    log(f"Reading metadata with a queue of {N_META_WORKERS} workers ...")
    rows, t0, n = [None] * len(entries), time.time(), len(entries)
    with mp.Pool(N_META_WORKERS) as pool:
        for k, (idx, row) in enumerate(pool.imap_unordered(
                extract_metadata, list(enumerate(entries)), chunksize=1), 1):
            rows[idx] = row
            st = row.get('status', '')
            if st == 'ok':
                status = f"{row.get('acq_center', '')}  corners: {row.get('corner_source', '')}"
            else:
                msg = row.get('error', '')
                status = f"{st.upper()}: {msg[:160] + ('...' if len(msg) > 160 else '')}"
            log(f"  [{k}/{n}] {row['product']}  {status}  [{time.time() - t0:.0f}s]")
    host = socket.gethostname()
    for r in rows:
        r['catalogue_host'] = host
        try:          # one row per dataset: size of each part, and the total
            sizes = [os.path.getsize(p) for p in r['all_parts'].split(';')]
            r['part_sizes'] = ';'.join(str(s) for s in sizes)
            r['zip_size'] = str(sum(sizes))
        except OSError:
            r['part_sizes'] = r['zip_size'] = ''
    sort_rows(rows)
    return rows


# ====================================== scenes, representatives, --verify
UNDAMAGED = ('ok', 'no_footprint')            # zip structure checked and sound
DAMAGED = ('damaged_recovered', 'damaged_unreadable', 'error')
DERIVED_COLUMNS = ('source_csv', 'n_copies', 'n_copies_ok', 'n_copies_damaged',
                   'scene_key', 'is_representative', 'scene_state',
                   'representative_path', 'representative_host', 'verify_status')


def scene_key(r):
    """Identity of a scene/product, the same for every copy (damaged or not):
    the zip name without extension or part number."""
    return r.get('product') or os.path.basename(r.get('zip_path', ''))


def row_parts(r):
    return [p for p in r.get('all_parts', '').split(';') if p] or [r.get('zip_path', '')]


def is_local_copy(r, host):
    """The row's files are on this machine and still have the recorded size."""
    if r.get('catalogue_host', '') not in ('', host):
        return False
    try:
        size = sum(os.path.getsize(p) for p in row_parts(r))
    except OSError:
        return False
    return not r.get('zip_size') or str(size) == r['zip_size']


def verify_entry(job):
    """Worker for --verify: read every byte of every file in the zip(s) and
    compare its CRC-32 and size with those stored in the zip directory.
    Returns (job index, verify_status text)."""
    idx, entry = job
    t0 = time.time()
    try:
        a = Archive(entry)
    except ArchiveError as e:
        return idx, f"FAILED: unreadable zip ({e})"
    try:
        if a.damage:
            return idx, f"FAILED: damaged zip structure ({a.damage})"
        n_files, n_bytes = 0, 0
        for r in a.readers:
            if isinstance(r, zipfile.ZipFile):
                items = [(i.filename, i.CRC, i.file_size, lambda i=i, r=r: r.open(i))
                         for i in r.infolist() if not i.is_dir()]
            else:
                items = [(n, r.crcs.get(n), m[3], None) for n, m in r.members.items()
                         if not n.endswith('/')]
            for name, want_crc, want_size, opener in items:
                crc, size = 0, 0
                if opener:
                    with opener() as fh:
                        while True:
                            b = fh.read(1 << 22)
                            if not b:
                                break
                            crc, size = zlib.crc32(b, crc), size + len(b)
                else:
                    for b in r.iter_member(name):
                        crc, size = zlib.crc32(b, crc), size + len(b)
                if size != want_size:
                    return idx, f"FAILED: {name} is {size:,} bytes, the zip says {want_size:,}"
                if want_crc is not None and crc != want_crc:
                    return idx, f"FAILED: CRC mismatch in {name} (data corrupted)"
                n_files, n_bytes = n_files + 1, n_bytes + size
        return idx, (f"passed: {n_files} file(s), {fmt_size(n_bytes).split(' (')[0]} checked "
                     f"on {socket.gethostname()} in {time.time() - t0:.0f}s")
    except Exception as e:
        return idx, f"FAILED: {type(e).__name__}: {e}"
    finally:
        a.close()


def choose_representatives(rows, verify=False):
    """Group copies by scene_key and mark one representative per scene:
    passed --verify > undamaged > status unknown (older CSV) > damaged;
    then the size most undamaged copies share, a copy on this machine, path.
    With verify=True, the best local undamaged copy of each scene is CRC-
    checked; if it fails the next copy is checked, until one passes or none
    is left. Earlier verify results (e.g. from another machine) are kept."""
    host = socket.gethostname()
    groups = {}
    for r in rows:
        r['scene_key'] = scene_key(r)
        groups.setdefault(r['scene_key'], []).append(r)

    def size_mode(g):
        sizes = sorted(r['zip_size'] for r in g if r.get('zip_size')
                       and r.get('status', '') in UNDAMAGED + ('',)
                       and not r.get('verify_status', '').startswith('FAILED'))
        return max(sizes, key=sizes.count) if sizes else ''

    def rank(r, mode):
        v, st = r.get('verify_status', ''), r.get('status', '')
        if v.startswith('FAILED'):
            sr = 5
        else:
            sr = 0 if st in UNDAMAGED else 1 if st == '' else 3 if st == 'damaged_recovered' else 4
        vr = 0 if v.startswith('passed') else 1
        return (sr, vr, r.get('zip_size', '') != mode,
                r.get('catalogue_host', '') not in ('', host), r.get('zip_path', ''))

    if verify:
        rnd = 0
        while True:
            jobs = []
            for key, g in groups.items():
                if any(r.get('verify_status', '').startswith('passed') for r in g):
                    continue
                mode = size_mode(g)
                cands = sorted((r for r in g if r.get('status', '') in UNDAMAGED + ('',)
                                and not r.get('verify_status') and is_local_copy(r, host)),
                               key=lambda r: rank(r, mode))
                if cands:
                    jobs.append(cands[0])
            if not jobs:
                break
            rnd += 1
            total = sum(int(r.get('zip_size') or 0) for r in jobs)
            log(f"\nVerifying {len(jobs)} copy(ies), {fmt_size(total).split(' (')[0]} "
                f"(round {rnd}; every byte is read and CRC-checked) ...")
            t0 = time.time()
            entries = [(i, row_parts(r) if len(row_parts(r)) > 1 else row_parts(r)[0])
                       for i, r in enumerate(jobs)]
            with mp.Pool(N_META_WORKERS) as pool:
                for k, (i, res) in enumerate(pool.imap_unordered(verify_entry, entries, chunksize=1), 1):
                    jobs[i]['verify_status'] = res
                    log(f"  [{k}/{len(jobs)}] {jobs[i].get('product', '')}  "
                        f"{res[:150]}  [{time.time() - t0:.0f}s]")

    for key, g in groups.items():
        mode = size_mode(g)
        rep = min(g, key=lambda r: rank(r, mode))
        n_ok = sum(1 for r in g if r.get('status', '') in UNDAMAGED
                   and not r.get('verify_status', '').startswith('FAILED'))
        n_bad = sum(1 for r in g if r.get('status', '') in DAMAGED
                    or r.get('verify_status', '').startswith('FAILED'))
        st, v = rep.get('status', ''), rep.get('verify_status', '')
        if st in DAMAGED or v.startswith('FAILED'):
            state = 'NEEDS RE-DOWNLOAD: no undamaged copy found'
        elif st == 'not_jaxa_product':
            state = 'not a JAXA product'
        elif st == '':
            state = 'unknown (CSV from an older version of this script)'
        elif v.startswith('passed'):
            state = 'ok, verified'
        elif verify:
            state = ('ok, not verified: no copy on this machine'
                     if not is_local_copy(rep, host) else 'ok, not verified')
        else:
            state = 'ok'
        good_sizes = {r['zip_size'] for r in g if r.get('zip_size')
                      and r.get('status', '') in UNDAMAGED
                      and not r.get('verify_status', '').startswith('FAILED')}
        if len(good_sizes) > 1:
            state += f"; SIZE CONFLICT: undamaged copies have {len(good_sizes)} different sizes"
        site = format_sites(scene_sites(g)[0])
        for r in g:
            r['site_names'] = site
            r['is_representative'] = 'yes' if r is rep else 'no'
            r['scene_state'] = state
            r['n_copies'], r['n_copies_ok'], r['n_copies_damaged'] = str(len(g)), str(n_ok), str(n_bad)
            r['representative_path'] = rep.get('zip_path', '')
            r['representative_host'] = rep.get('catalogue_host', '')


def report_scenes(rows):
    """Scenes by state, then every scene that needs attention with its copies."""
    reps = [r for r in rows if r.get('is_representative') == 'yes']
    n_multi = sum(1 for r in reps if int(r.get('n_copies', '1')) > 1)
    log(f"\nScenes: {len(reps)} unique scene(s) in {len(rows)} row(s); "
        f"{n_multi} scene(s) have more than one copy")
    counts = {}
    for r in reps:
        k = r['scene_state'].split(';')[0]
        counts[k] = counts.get(k, 0) + 1
    for k in sorted(counts, key=lambda k: (not k.startswith('ok'), k)):
        log(f"  {counts[k]:5d}  {k}")
    attention = [r for r in reps if r['scene_state'].startswith('NEEDS')
                 or 'CONFLICT' in r['scene_state']]
    if attention:
        log(f"\nScenes needing attention ({len(attention)}):")
        by_key = {}
        for r in rows:
            by_key.setdefault(r['scene_key'], []).append(r)
        for rep in attention:
            log(f"  {rep['scene_key']}: {rep['scene_state']}")
            for r in by_key[rep['scene_key']]:
                where = r.get('catalogue_host') or r.get('source_csv', '')
                log(f"      {r.get('status', '') or '(no status)':18s} {r.get('zip_size', '?'):>14} B  "
                    f"{where}:{r.get('zip_path', '')}"
                    + (f"  [{r['verify_status'][:60]}]" if r.get('verify_status') else ''))


STATUS_MEANING = {
    'ok': 'metadata and footprint read',
    'no_footprint': 'JAXA metadata read, but no corner coordinates found',
    'not_jaxa_product': 'readable zip, but it holds no JAXA product files',
    'damaged_recovered': 'broken zip structure; metadata read from the intact files',
    'damaged_unreadable': 'broken zip; nothing could be read (see diagnosis)',
    'error': 'unexpected error while reading (see error)',
    '': 'no status (CSV from an older version of this script)',
}


def report_status(rows):
    """Counts per status, then one line per product that is not 'ok'."""
    counts = {}
    for r in rows:
        counts[r.get('status', '')] = counts.get(r.get('status', ''), 0) + 1
    log("\nStatus of the catalogued products:")
    for st in sorted(counts, key=lambda s: (s != 'ok', s)):
        log(f"  {counts[st]:5d}  {st or '(none)':20s} {STATUS_MEANING.get(st, '')}")
    bad = [r for r in rows if r.get('status', '') not in ('ok', '')]
    if bad:
        log(f"\nProducts needing attention ({len(bad)}); full text in the error and "
            f"diagnosis columns of the CSV:")
        for r in bad:
            log(f"  {r.get('status', ''):18s} {r.get('zip_path', '')}")
            if r.get('is_representative') == 'no' and not r.get('scene_state', '').startswith('NEEDS'):
                log(f"  {'':18s} (scene is fine: representative copy is "
                    f"{r.get('representative_host', '')}:{r.get('representative_path', '')})")
            log(f"  {'':18s} {r.get('error', '')[:300]}")
            for line in [x for x in r.get('diagnosis', '').split(' | ') if 'CONCLUSION' in x]:
                log(f"  {'':18s} {line}")


def dataset_key(r):
    """Identity of a dataset independent of where (and how well) it is
    stored: the scene key (zip name without extension / part number)."""
    return r.get('scene_key') or scene_key(r)


def fmt_size(n):
    for unit, f in (('TB', 1e12), ('GB', 1e9), ('MB', 1e6)):
        if n >= f:
            return f"{n / f:,.2f} {unit} ({n:,} bytes)"
    return f"{n:,} bytes"


def report_sizes(rows):
    """Print two totals over all data files (each part of a multipart zip is
    a file): space used (each file once per machine + path, so copies in
    different places all count) and unique data (identical files, i.e. same
    name and size, counted once however many copies exist)."""
    by_location, by_file, no_size = {}, {}, 0
    for r in rows:
        parts = [p for p in r.get('all_parts', '').split(';') if p] or [r.get('zip_path', '')]
        sizes = [s for s in r.get('part_sizes', '').split(';') if s]
        if len(sizes) != len(parts) and len(parts) == 1 and r.get('zip_size'):
            sizes = [r['zip_size']]
        if len(sizes) != len(parts) or not all(s.isdigit() for s in sizes):
            no_size += 1
            continue
        where = r.get('catalogue_host') or os.path.dirname(r.get('source_csv', '')) or '?'
        for p, s in zip(parts, sizes):
            by_location[(where, p)] = int(s)
            by_file[(os.path.basename(p), int(s))] = int(s)
    log(f"\nData file sizes ({len(by_location)} file(s) counting every copy, "
        f"{len(by_file)} unique file(s)):")
    log(f"  space used (every copy counted):     {fmt_size(sum(by_location.values()))}")
    log(f"  unique data (each file counted once): {fmt_size(sum(by_file.values()))}")
    if no_size:
        log(f"  [WARN] {no_size} row(s) have no size recorded (CSV from an older version "
            f"of this script) and are not in these totals")


def sort_rows(rows):
    rows.sort(key=lambda r: (r.get('satellite', ''), r.get('acq_center', ''),
                             r.get('catalogue_host', ''), r.get('zip_path', '')))


def upgrade_old_corners(r):
    """CSVs from the first version stored image-order corners (lt/rt/lb/rb);
    convert them to the geographic NW/NE/SW/SE columns used now."""
    if to_float(r.get('nw_lat')) is not None:
        return
    old = [(to_float(r.get(c + '_lat')), to_float(r.get(c + '_lon'))) for c in ('lt', 'rt', 'lb', 'rb')]
    if all(a is not None and b is not None for a, b in old):
        for name, (lat, lon) in geo_corners(old).items():
            r[name + '_lat'], r[name + '_lon'] = f"{lat:.7f}", f"{lon:.7f}"
        r['corner_source'] = (r.get('corner_source', '') + ' (from old lt/rt/lb/rb columns)').strip()
        for c in ('lt', 'rt', 'lb', 'rb'):
            r.pop(c + '_lat', None)
            r.pop(c + '_lon', None)


def stage_merge():
    """--merge: combine every JAXA_products_catalogue.csv found below the
    present folder (any depth) into one list of rows. Rows that are exact
    repeats of each other (e.g. the same CSV copied twice) are kept once."""
    root = os.getcwd()
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if CSV_FILE in filenames:
            found.append(os.path.join(dirpath, CSV_FILE))
    if not found:
        log(f"[ERROR] no {CSV_FILE} found anywhere below {root}")
        return []

    log(f"Merging {len(found)} catalogue(s) found below {root}:")
    rows, seen, n_repeat = [], {}, 0
    for path in found:
        rel = os.path.relpath(path, root)
        try:
            with open(path, newline='', encoding='utf-8', errors='surrogateescape') as f:
                reader = csv.DictReader(f)
                if 'zip_path' not in (reader.fieldnames or []):
                    log(f"  [WARN] skipped {rel}: not a jaxa_catalogue.py CSV (no zip_path column)")
                    continue
                n_in = n_new = 0
                for r in reader:
                    n_in += 1
                    r = {k: v for k, v in r.items()
                         if isinstance(k, str) and isinstance(v, str) and v != ''}
                    upgrade_old_corners(r)
                    key = tuple(sorted((k, v) for k, v in r.items()
                                       if k not in DERIVED_COLUMNS))
                    if key in seen:
                        n_repeat += 1
                        kept = seen[key]          # keep a verify result if either has one
                        if r.get('verify_status') and not kept.get('verify_status', '').startswith('passed'):
                            kept['verify_status'] = r['verify_status']
                        continue
                    seen[key] = r
                    r.setdefault('source_csv', rel)   # keep the origin of a re-merged row
                    rows.append(r)
                    n_new += 1
        except (OSError, csv.Error) as e:
            log(f"  [WARN] skipped {rel}: {e}")
            continue
        log(f"  {rel}: {n_in} row(s), {n_new} added")
    if n_repeat:
        log(f"  {n_repeat} row(s) were exact repeats of rows already added and were kept once")
    if not rows:
        log("[ERROR] the catalogue CSVs found hold no rows")
        return []
    log(f"\nMerged {len(rows)} row(s) from {len(found)} catalogue(s)")
    return rows


def write_csv(rows, path):
    extra = sorted({k for r in rows for k in r} - set(STD_COLUMNS))
    columns = STD_COLUMNS + extra
    tmp = path + '.partial'
    with open(tmp, 'w', newline='', encoding='utf-8', errors='surrogateescape') as f:
        w = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        w.writerow(columns)
        for r in rows:
            w.writerow([clean(str(r.get(c, ''))) for c in columns])
    os.replace(tmp, path)        # only a complete CSV ever has the final name
    log(f"\nWrote {path}: {len(rows)} product(s), {len(columns)} column(s)")


# ============================================================ stage 2: shp
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def corners_match(a, b, km):
    """Every corner within km of the matching corner (lt-lt, rt-rt, ...)."""
    return all(haversine_km(a[i], a[i + 1], b[i], b[i + 1]) <= km for i in range(0, 8, 2))


def equivalence_classes(polys, km):
    """Union-find over polygons (8-tuples lat,lon x4). Returns list of lists
    of polygon indices. Candidate pairs are pruned by the NW-corner latitude:
    a corner distance <= km implies a latitude difference <= km / R."""
    parent = list(range(len(polys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    max_dlat = math.degrees(km / EARTH_RADIUS_KM) * 1.000001
    order = sorted(range(len(polys)), key=lambda i: polys[i][0])
    for a_pos, i in enumerate(order):
        for j in order[a_pos + 1:]:
            if polys[j][0] - polys[i][0] > max_dlat:
                break
            if find(i) != find(j) and corners_match(polys[i], polys[j], km):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(polys)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def ring_of(c):
    """8-tuple (nw_lat, nw_lon, ne_lat, ne_lon, sw_lat, sw_lon, se_lat, se_lon)
    -> closed ring of (lon, lat) going NW, NE, SE, SW."""
    pts = [(c[1], c[0]), (c[3], c[2]), (c[7], c[6]), (c[5], c[4])]
    return pts + [pts[0]]


def join_unique(values, limit=254):
    s = ';'.join(sorted({v for v in values if v}))
    return s[:limit]


def write_shapefile(path, fields, features, points=False):
    """fields: [(name<=10 chars, 'str'|'int'|'real')];
    features: [(ring [(lon, lat), ...], {field: value})], or with
    points=True: [((lon, lat), {field: value})]."""
    from osgeo import ogr, osr
    ogr.UseExceptions()
    drv = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(path):
        drv.DeleteDataSource(path)
    ds = drv.CreateDataSource(path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    lyr = ds.CreateLayer(os.path.splitext(os.path.basename(path))[0], srs,
                         ogr.wkbPoint if points else ogr.wkbPolygon,
                         options=['ENCODING=UTF-8'])
    kinds = {'str': ogr.OFTString, 'int': ogr.OFTInteger, 'real': ogr.OFTReal}
    for name, kind in fields:
        fd = ogr.FieldDefn(name, kinds[kind])
        if kind == 'str':
            fd.SetWidth(254)
        lyr.CreateField(fd)
    for coords, attrs in features:
        if points:
            poly = ogr.Geometry(ogr.wkbPoint)
            poly.AddPoint_2D(coords[0], coords[1])
        else:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for lon, lat in coords:
                ring.AddPoint_2D(lon, lat)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
        feat = ogr.Feature(lyr.GetLayerDefn())
        for k, v in attrs.items():
            if v is None or v == '':
                continue
            if isinstance(v, str):
                v = v[:254]
            feat.SetField(k, v)
        feat.SetGeometry(poly)
        lyr.CreateFeature(feat)
    ds = None


# ======================================================= site names (AOIs)
def site_names_in_path(path):
    """Site names from folder names like 2021_sparks_lake or
    2017_ELEPHANT_HILL (SITE_FOLDER_REGEX), skipping generic folders such as
    2025_JAXA_DATA or 2023_EODMS_data_from_Hao (SITE_STOPWORDS).
    Returns [(display name, year), ...]."""
    out = []
    named = {k.lower(): v for k, v in FOLDER_SITE_NAMES.items()}
    for comp in os.path.dirname(path).replace('\\', '/').split('/'):
        if comp.lower() in named:
            out.append((named[comp.lower()], ''))
            continue
        m = SITE_FOLDER_REGEX.match(comp)
        if not m:
            continue
        tokens = [t for t in re.split(r'[_\-\s.]+', m.group(2).lower()) if t]
        if not tokens or all(t.isdigit() for t in tokens):
            continue
        if any(t in SITE_STOPWORDS for t in tokens):
            continue
        out.append((' '.join(t.capitalize() for t in tokens), m.group(1)))
    return out


def folder_hint(path):
    """The zip's own folder name unless it is just a number (order folder)."""
    d = os.path.basename(os.path.dirname(path))
    return '' if not d or d.isdigit() else d


def scene_sites(copies):
    """Site names found in the folders of all copies of one scene:
    {name: {years}}, plus a Counter of folder hints."""
    names, hints = {}, collections.Counter()
    for r in copies:
        for p in row_parts(r):
            for name, year in site_names_in_path(p):
                names.setdefault(name, set()).add(year)
            h = folder_hint(p)
            if h:
                hints[h] += 1
    return names, hints


def format_sites(names):
    """{name: {years}} -> 'Sparks Lake (2021); Sparks Tremont (2021)'."""
    return '; '.join(n + (f" ({', '.join(sorted(y - {''}))})" if y - {''} else '')
                     for n, y in sorted(names.items()))


def short_date_list(scenes):
    """'2015-08-01 FBD ALOS2 A, ...' for a list of catalogue rows."""
    return ', '.join(date_items(scenes))


def date_items(scenes):
    """['2015-08-01 FBD ALOS2 A', ...]: one entry per scene, in time order
    (date, mode, satellite, A/D orbit direction)."""
    items = []
    for r in sorted(scenes, key=lambda r: r.get('acq_center') or r.get('acq_start') or ''):
        d = (r.get('acq_center') or r.get('acq_start') or '????-??-??')[:10]
        o = {'ascending': 'A', 'descending': 'D'}.get(r.get('orbit_direction', ''), '')
        items.append(' '.join(x for x in (d, r.get('observation_mode') or '?',
                                          r.get('satellite', ''), o) if x))
    return items


def truncate_list(text, limit=254):
    """Fit a comma-separated list in a shapefile text field (254 chars)."""
    if len(text) <= limit:
        return text
    parts, out = text.split(', '), ''
    for k, p in enumerate(parts):
        tail = f" ... +{len(parts) - k} more (see {AREAS_CSV})"
        if len(out) + len(p) + 2 + len(tail) > limit:
            return out + tail
        out = p if not out else out + ', ' + p
    return out[:limit]


def poly_geom(ring):
    from osgeo import ogr
    rg = ogr.Geometry(ogr.wkbLinearRing)
    for lon, lat in ring:
        rg.AddPoint_2D(lon, lat)
    g = ogr.Geometry(ogr.wkbPolygon)
    g.AddGeometry(rg)
    return g


def write_geom_shapefile(path, fields, features):
    """Like write_shapefile, but takes OGR geometries (multipolygons)."""
    from osgeo import ogr, osr
    ogr.UseExceptions()
    drv = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(path):
        drv.DeleteDataSource(path)
    ds = drv.CreateDataSource(path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    lyr = ds.CreateLayer(os.path.splitext(os.path.basename(path))[0], srs,
                         ogr.wkbMultiPolygon, options=['ENCODING=UTF-8'])
    kinds = {'str': ogr.OFTString, 'int': ogr.OFTInteger, 'real': ogr.OFTReal}
    for name, kind in fields:
        fd = ogr.FieldDefn(name, kinds[kind])
        if kind == 'str':
            fd.SetWidth(40 if re.match(r'^DATE\d+$', name) else 254)
        lyr.CreateField(fd)
    for geom, attrs in features:
        feat = ogr.Feature(lyr.GetLayerDefn())
        for k, v in attrs.items():
            if v is None or v == '':
                continue
            feat.SetField(k, v[:254] if isinstance(v, str) else v)
        feat.SetGeometry(ogr.ForceToMultiPolygon(geom))
        lyr.CreateFeature(feat)
    ds = None


AREA_KML_STYLE = """    <Style id="area">
      <LineStyle><color>ff0000ff</color><width>2</width></LineStyle>
      <PolyStyle><color>330000ff</color><fill>1</fill><outline>1</outline></PolyStyle>
    </Style>"""


def write_area_kmz(path, geom, attrs, dates):
    """One area as a KMZ for Google Earth: the outline, and a pop-up with
    the area's attributes and every acquisition date with its mode."""
    from xml.sax.saxutils import escape
    keys = ['AREA_ID', 'NAME_STAT', 'N_SCENES', 'N_DATES', 'N_MODES', 'MODES', 'SAT_MODES',
            'SATS', 'ORB_DIRS', 'CLASS_IDS', 'FIRST_ACQ', 'LAST_ACQ', 'YEARS', 'N_COPIES',
            'N_REDOWNLD', 'HOSTS', 'SITES', 'CTR_LAT', 'CTR_LON']
    rows = ''.join(f"<tr><td><b>{k}</b></td><td>{escape(str(attrs.get(k, '')))}</td></tr>"
                   for k in keys if attrs.get(k, '') not in ('', None))
    dl = ''.join(f"<tr><td>{k}</td><td>{escape(d)}</td></tr>" for k, d in enumerate(dates, 1))
    html = (f"<table>{rows}</table><p><b>Acquisitions ({len(dates)}): date, mode, "
            f"satellite, orbit (A/D)</b></p><table>{dl}</table>")
    kml = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
           f'<kml xmlns="http://www.opengis.net/kml/2.2">\n  <Document>\n'
           f'    <name>{escape(attrs["AREA_ID"] + " " + attrs["NAME"])}</name>\n'
           f'{AREA_KML_STYLE}\n    <Placemark>\n      <name>{escape(attrs["NAME"])}</name>\n'
           f'      <description><![CDATA[{html}]]></description>\n'
           f'      <styleUrl>#area</styleUrl>\n      {geom.ExportToKML()}\n'
           f'    </Placemark>\n  </Document>\n</kml>\n')
    tmp = path + '.partial'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('doc.kml', kml.encode('utf-8'))
    os.replace(tmp, path)


def area_file_name(area_id, name):
    """'0013', 'Sparks Lake / Sparks Tremont' -> '0013_Sparks_Lake_Sparks_Tremont.shp'"""
    safe = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')[:60] or 'area'
    return f"{area_id}_{safe}.shp"


def surface_area(g):
    """Area of the polygon parts of a geometry; 0 for lines/points (footprints
    that only touch), without GDAL's 'non-surface geometry' warning."""
    if g is None:
        return 0.0
    name = g.GetGeometryName()
    if name in ('POLYGON', 'MULTIPOLYGON'):
        return g.GetArea()
    if name == 'GEOMETRYCOLLECTION':
        return sum(surface_area(g.GetGeometryRef(k)) for k in range(g.GetGeometryCount()))
    return 0.0


def group_areas(geoms, fraction):
    """Union-find over class polygons: linked when their overlap is at least
    `fraction` of the smaller polygon's area. Returns lists of indices."""
    parent = list(range(len(geoms)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(geoms)):
        for j in range(i + 1, len(geoms)):
            gi, gj = geoms[i], geoms[j]
            if not gi.Intersects(gj):
                continue
            small = min(gi.GetArea(), gj.GetArea())
            if small > 0 and surface_area(gi.Intersection(gj)) / small >= fraction:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(geoms)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def stage_shapefiles(csv_path):
    try:
        from osgeo import ogr  # noqa: F401
    except ImportError:
        log(f"\n[ERROR] no shapefiles made: this Python ({sys.executable}) has no GDAL "
            f"bindings (osgeo). {csv_path} is complete, so either\n"
            f"  * install them: `sudo apt install python3-gdal` (Ubuntu) or "
            f"`brew install gdal` (macOS), then run this script again here, or\n"
            f"  * copy {csv_path} to a machine that has GDAL and make the shapefiles "
            f"there with --merge")
        sys.exit(1)

    with open(csv_path, newline='', encoding='utf-8', errors='surrogateescape') as f:
        rows = list(csv.DictReader(f))

    # site names per scene, from the folders of ALL its copies (any machine)
    copies = {}
    for r in rows:
        copies.setdefault(dataset_key(r), []).append(r)
    sites = {k: scene_sites(v) for k, v in copies.items()}

    usable = []
    n_reps = sum(1 for r in rows if r.get('is_representative', 'yes') != 'no')
    for n, r in enumerate(rows, 1):
        if r.get('is_representative', 'yes') == 'no':   # one copy per scene
            continue
        c = tuple(to_float(r.get(k)) for k in CORNER_KEYS)
        if all(v is not None for v in c):
            usable.append((n, r, tuple(round(v, 7) for v in c)))
    n_centre_only = sum(1 for r in rows if to_float(r.get('center_lat')) is not None
                        and to_float(r.get('nw_lat')) is None
                        and r.get('is_representative', 'yes') != 'no')
    log(f"\nShapefiles from {csv_path}, one representative copy per scene: "
        f"{len(usable)} of {n_reps} scene(s) have 4 corners"
        + (f", {n_centre_only} more have only a scene centre" if n_centre_only else ""))
    if not usable:
        log("[WARN] no products with corner coordinates; only the scene centres are written")
        write_centres(rows, {}, {})
        return [SHP_CENTRES], {}, {}

    def acq(r):
        return r.get('acq_center') or r.get('acq_start') or ''

    # 1. unique exact polygons
    uniq = {}
    for n, r, c in usable:
        uniq.setdefault(c, []).append((n, r))
    polys = list(uniq.keys())

    # 2. equivalence classes (transitive closure of "all corners within km")
    classes = equivalence_classes(polys, CORNER_THRESHOLD_KM)

    def n_scenes(g):          # unique datasets (copies counted once)
        return len({dataset_key(r) for i in g for _, r in uniq[polys[i]]})

    classes.sort(key=lambda g: (-n_scenes(g), -sum(len(uniq[polys[i]]) for i in g),
                                min(acq(r) for i in g for _, r in uniq[polys[i]])))
    width = len(str(len(classes))) + 2           # at least two leading zeros
    class_of, class_members, class_geom, class_ids = {}, [], [], []
    for k, g in enumerate(classes, 1):
        cid = str(k).zfill(width)
        for i in g:
            class_of[i] = cid
        class_ids.append(cid)
        class_members.append([(n, r, polys[i]) for i in g for n, r in uniq[polys[i]]])
        mean = tuple(sum(polys[i][j] for i in g) / len(g) for j in range(8))
        class_geom.append(poly_geom(ring_of(mean)))

    # 6. areas: classes over the same place (any mode / satellite / direction)
    agroups = group_areas(class_geom, AREA_OVERLAP_FRACTION)
    agroups.sort(key=lambda a: (-sum(len(class_members[c]) for c in a),
                                min(acq(r) for c in a for _, r, _ in class_members[c])))
    awidth = len(str(len(agroups))) + 2
    areas, area_of_class = [], {}
    for k, a in enumerate(agroups, 1):
        aid = str(k).zfill(awidth)
        geom = class_geom[a[0]].Clone()
        for c in a[1:]:
            geom = geom.Union(class_geom[c])
        scenes = [r for c in a for _, r, _ in class_members[c]]
        names, hints = {}, collections.Counter()
        for r in scenes:
            nm, h = sites[dataset_key(r)]
            for name, years in nm.items():
                names.setdefault(name, set()).update(years)
            hints.update(h)
        areas.append({'id': aid, 'classes': sorted(a), 'geom': geom, 'scenes': scenes,
                      'names': names, 'hints': hints})
        for c in a:
            area_of_class[c] = aid

    # area names: manual sites first, then folder names; flag the rest
    name_users = collections.defaultdict(list)
    for ar in areas:
        manual = [nm for nm, lat, lon in MANUAL_SITES
                  if ar['geom'].Contains(ogr.CreateGeometryFromWkt(f"POINT ({lon} {lat})"))]
        c = ar['geom'].Centroid()
        if manual:
            ar['name'], ar['status'] = ' / '.join(manual), 'manual (MANUAL_SITES)'
        elif len(ar['names']) == 1:
            ar['name'], ar['status'] = next(iter(ar['names'])), 'resolved'
        elif ar['names']:
            ranked = sorted(ar['names'], key=lambda nm: (-sum(
                1 for r in ar['scenes'] if nm in sites[dataset_key(r)][0]), nm))
            ar['name'], ar['status'] = ' / '.join(ranked), 'several names'
        else:
            hint = ar['hints'].most_common(1)[0][0] if ar['hints'] else \
                f"{c.GetY():.2f}N {abs(c.GetX()):.2f}{'W' if c.GetX() < 0 else 'E'}"
            ar['name'], ar['status'] = f"UNRESOLVED ({hint})", 'unresolved'
        for nm in (ar['names'] if not manual else manual):
            name_users[nm].append(ar['id'])
    for ar in areas:
        shared = sorted({other for nm in ar['names'] for other in name_users[nm]
                         if other != ar['id']})
        ar['note'] = (f"name also used by area(s) {', '.join(shared)}" if shared else '')

    area_by_id = {ar['id']: ar for ar in areas}
    row_class = {n: class_of[i] for i, c in enumerate(polys) for n, _ in uniq[c]}
    row_area = {n: area_of_class[class_ids.index(cid)] for n, cid in row_class.items()}

    def area_name_of_class(ci):
        return area_by_id[area_of_class[ci]]['name']

    # 1. unique polygons
    feats = []
    for i, c in enumerate(polys):
        members = uniq[c]
        ci = class_ids.index(class_of[i])
        dates = sorted(acq(r) for _, r in members if acq(r))
        feats.append((ring_of(c), {
            'NAME': area_name_of_class(ci), 'AREA_ID': area_of_class[ci],
            'U_ID': i + 1, 'CLASS_ID': class_of[i], 'N_SCENES': len(members),
            'SATS': join_unique(r.get('satellite') for _, r in members),
            'FIRST_ACQ': dates[0] if dates else '', 'LAST_ACQ': dates[-1] if dates else '',
            'ORB_DIR': join_unique(r.get('orbit_direction') for _, r in members),
            'MODES': join_unique(r.get('observation_mode') for _, r in members),
            'POLS': join_unique(r.get('polarizations') for _, r in members)}))
    write_shapefile(SHP_UNIQUE, [('NAME', 'str'), ('AREA_ID', 'str'), ('U_ID', 'int'),
                                 ('CLASS_ID', 'str'), ('N_SCENES', 'int'),
                                 ('SATS', 'str'), ('FIRST_ACQ', 'str'), ('LAST_ACQ', 'str'),
                                 ('ORB_DIR', 'str'), ('MODES', 'str'), ('POLS', 'str')], feats)
    log(f"  1. {SHP_UNIQUE}: {len(polys)} unique polygon(s)")

    # 3. one shapefile per class, one polygon per scene
    member_fields = [('NAME', 'str'), ('SITE', 'str'), ('AREA_ID', 'str'), ('CLASS_ID', 'str'),
                     ('CSV_ROW', 'int'), ('ACQ_DATE', 'str'), ('SAT', 'str'), ('MODE', 'str'),
                     ('ORB_DIR', 'str'), ('LOOK', 'str'), ('POL', 'str'), ('LEVEL', 'str'),
                     ('SCENE_ID', 'str'), ('PROD_ID', 'str'),
                     ('ACQ_START', 'str'), ('ACQ_CENTER', 'str'), ('INC_ANG', 'real'),
                     ('OFFNADIR', 'real'), ('STATUS', 'str'), ('STATE', 'str'),
                     ('N_COPIES', 'real'), ('REP_HOST', 'str'), ('ZIP', 'str')]
    os.makedirs(CLASS_DIR, exist_ok=True)
    for old in os.listdir(CLASS_DIR):                   # clear the previous run
        if re.match(r'^\d+_members\.(shp|shx|dbf|prj|cpg)$', old):
            os.remove(os.path.join(CLASS_DIR, old))
    class_feats = []
    for ci, cid in enumerate(class_ids):
        members = class_members[ci]
        scenes = [r for _, r, _ in members]
        dates = sorted(acq(r) for r in scenes if acq(r))
        fname = f"{cid}_members.shp"
        modes = sorted({r.get('observation_mode', '') for r in scenes} - {''})
        mean = tuple(sum(polys[i][j] for i in classes[ci]) / len(classes[ci]) for j in range(8))
        class_feats.append((ring_of(mean), {
            'NAME': area_name_of_class(ci), 'AREA_ID': area_of_class[ci], 'CLASS_ID': cid,
            'N_SCENES': n_scenes(classes[ci]), 'N_POLYS': len(classes[ci]),
            'N_MODES': len(modes), 'MODES': ','.join(modes),
            'SATS': join_unique(r.get('satellite') for r in scenes),
            'ORB_DIR': join_unique(r.get('orbit_direction') for r in scenes),
            'LOOK': join_unique(r.get('look_direction') for r in scenes),
            'FIRST_ACQ': dates[0] if dates else '', 'LAST_ACQ': dates[-1] if dates else '',
            'DATES': truncate_list(short_date_list(scenes)),
            'FILE': os.path.join(CLASS_DIR, fname)}))
        mfeats = []
        for n, r, c in sorted(members, key=lambda t: acq(t[1])):
            mfeats.append((ring_of(c), {
                'NAME': area_name_of_class(ci),
                'SITE': format_sites(sites[dataset_key(r)][0]) or '(none in folder names)',
                'AREA_ID': area_of_class[ci], 'CLASS_ID': cid, 'CSV_ROW': n,
                'ACQ_DATE': acq(r)[:10], 'SAT': r.get('satellite', ''),
                'MODE': r.get('observation_mode', ''), 'ORB_DIR': r.get('orbit_direction', ''),
                'LOOK': r.get('look_direction', ''), 'POL': r.get('polarizations', ''),
                'LEVEL': r.get('processing_level', ''),
                'SCENE_ID': r.get('scene_id', ''), 'PROD_ID': r.get('product_id', ''),
                'ACQ_START': r.get('acq_start', ''), 'ACQ_CENTER': r.get('acq_center', ''),
                'INC_ANG': to_float(r.get('incidence_angle_center')),
                'OFFNADIR': to_float(r.get('off_nadir_angle')),
                'STATUS': r.get('status', ''), 'STATE': r.get('scene_state', '')[:254],
                'N_COPIES': to_float(r.get('n_copies')),
                'REP_HOST': r.get('catalogue_host', ''), 'ZIP': r.get('zip_path', '')}))
        write_shapefile(os.path.join(CLASS_DIR, fname), member_fields, mfeats)

    # 2. classes (fine level: same footprint within CORNER_THRESHOLD_KM)
    write_shapefile(SHP_CLASSES, [('NAME', 'str'), ('AREA_ID', 'str'), ('CLASS_ID', 'str'),
                                  ('N_SCENES', 'int'), ('N_POLYS', 'int'),
                                  ('N_MODES', 'int'), ('MODES', 'str'), ('SATS', 'str'),
                                  ('ORB_DIR', 'str'), ('LOOK', 'str'),
                                  ('FIRST_ACQ', 'str'), ('LAST_ACQ', 'str'),
                                  ('DATES', 'str'), ('FILE', 'str')], class_feats)
    log(f"  2. {SHP_CLASSES}: {len(classes)} equivalence class(es) "
        f"(corner threshold {CORNER_THRESHOLD_KM:g} km)")
    log(f"  3. {CLASS_DIR}/: {len(classes)} shapefile(s), "
        f"{class_ids[0]}_members.shp ... {class_ids[-1]}_members.shp")

    write_centres(rows, row_class, {n: (row_area[n], area_by_id[row_area[n]]['name'])
                                    for n in row_area})

    # 6. areas (top level: one polygon per place, all modes together)
    afeats, csv_rows = [], []
    for ar in areas:
        sc = ar['scenes']
        dates = sorted(acq(r) for r in sc if acq(r))
        modes = sorted({r.get('observation_mode', '') for r in sc} - {''})
        satmodes = sorted({f"{r.get('satellite', '')} {r.get('observation_mode', '')}".strip()
                           for r in sc})
        sats = sorted({r.get('satellite', '') for r in sc} - {''})
        years = sorted({d[:4] for d in dates})
        n_copies = sum(int(to_float(r.get('n_copies')) or 1) for r in sc)
        n_redl = sum(1 for r in sc if r.get('scene_state', '').startswith('NEEDS'))
        hosts = sorted({c.get('catalogue_host', '') for r in sc
                        for c in copies[dataset_key(r)]} - {''})
        c = ar['geom'].Centroid()
        datelist = short_date_list(sc)
        per_scene = date_items(sc)
        attrs = {
            'NAME': ar['name'], 'NAME_STAT': ar['status'], 'NAME_NOTE': ar['note'],
            'AREA_ID': ar['id'], 'N_SCENES': len(sc), 'N_DATES': len({d[:10] for d in dates}),
            'N_MODES': len(modes), 'MODES': ','.join(modes), 'SAT_MODES': ','.join(satmodes),
            'N_SATS': len(sats), 'SATS': ','.join(sats),
            'ORB_DIRS': ','.join(sorted({r.get('orbit_direction', '') for r in sc} - {''})),
            'N_CLASSES': len(ar['classes']),
            'CLASS_IDS': ','.join(class_ids[ci] for ci in ar['classes']),
            'FIRST_ACQ': dates[0][:10] if dates else '', 'LAST_ACQ': dates[-1][:10] if dates else '',
            'YEARS': ','.join(years), 'N_COPIES': n_copies, 'N_REDOWNLD': n_redl,
            'HOSTS': ','.join(hosts), 'SITES': format_sites(ar['names']),
            'NAME_HINT': ', '.join(h for h, _ in ar['hints'].most_common(3)),
            'CTR_LAT': round(c.GetY(), 4), 'CTR_LON': round(c.GetX(), 4),
            'FILE': os.path.join(AREAS_DIR, area_file_name(ar['id'], ar['name'])),
            'DATES': truncate_list(datelist)}
        attrs['_items'] = per_scene
        afeats.append((ar['geom'], attrs))
        csv_rows.append(dict(attrs, DATES=datelist))
    afields = [('NAME', 'str'), ('NAME_STAT', 'str'), ('NAME_NOTE', 'str'), ('AREA_ID', 'str'),
               ('N_SCENES', 'int'), ('N_DATES', 'int'), ('N_MODES', 'int'), ('MODES', 'str'),
               ('SAT_MODES', 'str'), ('N_SATS', 'int'), ('SATS', 'str'), ('ORB_DIRS', 'str'),
               ('N_CLASSES', 'int'), ('CLASS_IDS', 'str'), ('FIRST_ACQ', 'str'),
               ('LAST_ACQ', 'str'), ('YEARS', 'str'), ('N_COPIES', 'int'), ('N_REDOWNLD', 'int'),
               ('HOSTS', 'str'), ('SITES', 'str'), ('NAME_HINT', 'str'),
               ('CTR_LAT', 'real'), ('CTR_LON', 'real'), ('FILE', 'str'), ('DATES', 'str')]
    # DATE001, DATE002, ...: each scene's piece of DATES, in time order, as its
    # own field (as many fields as the largest area has scenes; a shapefile
    # holds at most 255 fields)
    max_n = max(len(a['_items']) for _, a in afeats)
    n_fields = min(max_n, 255 - len(afields))
    dw = max(3, len(str(n_fields)))
    date_fields = [(f"DATE{k:0{dw}d}", 'str') for k in range(1, n_fields + 1)]
    for _, a in afeats:
        for k, item in enumerate(a.pop('_items')[:n_fields], 1):
            a[f"DATE{k:0{dw}d}"] = item
    for cr, (_, a) in zip(csv_rows, afeats):
        cr.pop('_items', None)
        cr.update({k: a.get(k, '') for k, _ in date_fields})
    if n_fields < max_n:
        log(f"  [WARN] {SHP_AREAS}: only DATE{1:0{dw}d}..DATE{n_fields:0{dw}d} fit (shapefile "
            f"field limit); the largest area has {max_n} scenes, all listed in {AREAS_CSV}")
    afields = afields + date_fields
    write_geom_shapefile(SHP_AREAS, afields, afeats)

    # 8. one shapefile per area (same fields), e.g. JAXA_areas/0009_Elephant_Hill.shp
    os.makedirs(AREAS_DIR, exist_ok=True)
    for old in os.listdir(AREAS_DIR):                   # clear the previous run
        if re.match(r'^\d+_.*\.(shp|shx|dbf|prj|cpg|kmz)$', old):
            os.remove(os.path.join(AREAS_DIR, old))
    for (geom, a), cr in zip(afeats, csv_rows):
        write_geom_shapefile(a['FILE'], afields, [(geom, a)])
        write_area_kmz(a['FILE'][:-4] + '.kmz', geom, a, cr['DATES'].split(', '))
    unresolved = [(g, a) for g, a in afeats if a['NAME_STAT'] == 'unresolved']
    if unresolved:
        write_geom_shapefile(SHP_UNRESOLVED, afields, unresolved)
    else:
        from osgeo import ogr as _ogr
        if os.path.exists(SHP_UNRESOLVED):
            _ogr.GetDriverByName('ESRI Shapefile').DeleteDataSource(SHP_UNRESOLVED)
    tmp = AREAS_CSV + '.partial'
    with open(tmp, 'w', newline='', encoding='utf-8', errors='surrogateescape') as f:
        w = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        w.writerow([k for k, _ in afields])
        for cr in csv_rows:
            w.writerow([cr.get(k, '') for k, _ in afields])
    os.replace(tmp, AREAS_CSV)
    log(f"  6. {SHP_AREAS}: {len(areas)} area(s), one polygon per place with every mode "
        f"(classes overlapping >= {AREA_OVERLAP_FRACTION:.0%} of the smaller are joined); "
        f"full date lists in {AREAS_CSV}")
    log(f"  8. {AREAS_DIR}/: {len(afeats)} shapefile(s) and {len(afeats)} KMZ file(s), one "
        f"of each per area, e.g. {os.path.basename(afeats[0][1]['FILE'])[:-4]}.shp/.kmz")
    log(f"  7. {SHP_UNRESOLVED}: " + (f"{len(unresolved)} area(s) still without a name"
                                     if unresolved else "not written: every area has a name"))

    log(f"\nAreas (named from folder names; '{'UNRESOLVED'}' = no site folder found):")
    for cr in csv_rows:
        flag = '' if cr['NAME_STAT'] == 'resolved' else f"  [{cr['NAME_STAT']}]"
        log(f"  {cr['AREA_ID']}  {cr['NAME']}{flag}")
        log(f"  {'':{len(cr['AREA_ID'])}}  {cr['N_SCENES']} scene(s), {cr['N_DATES']} date(s), "
            f"{cr['FIRST_ACQ']}..{cr['LAST_ACQ']}, modes {cr['SAT_MODES']}, "
            f"classes {cr['CLASS_IDS']}" + (f"; {cr['NAME_NOTE']}" if cr['NAME_NOTE'] else ''))
    return [SHP_AREAS] + ([SHP_UNRESOLVED] if unresolved else []) + [SHP_CENTRES], row_class, {n: (a, area_by_id[a]['name'])
                                                 for n, a in row_area.items()}


def write_summary(csv_path, row_class, row_area):
    """JAXA_products_summary.csv: one row per scene (representative copy),
    grouped by area, then by equivalence class (most scenes first), with
    columns in front saying which area / class / site each scene belongs to.
    Scenes without corners (no class) come last."""
    with open(csv_path, newline='', encoding='utf-8', errors='surrogateescape') as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    copies = {}
    for r in rows:
        copies.setdefault(dataset_key(r), []).append(r)
    groups = {}
    for n, r in enumerate(rows, 1):
        if r.get('is_representative', 'yes') != 'no':     # one row per scene
            aid, aname = row_area.get(n, ('', ''))
            groups.setdefault((aid, row_class.get(n, '')), []).append(r)
    scenes = {k: len({dataset_key(r) for r in g}) for k, g in groups.items()}
    area_n = collections.Counter()
    for (aid, _), n in scenes.items():
        area_n[aid] += n
    names = {}
    for n, (aid, aname) in row_area.items():
        names[aid] = aname
    order = sorted(groups, key=lambda k: (k[0] == '', k[0], -scenes[k], k[1]))

    front = ['area_id', 'area_name', 'scene_site', 'class_id', 'class_shapefile',
             'class_unique_scenes', 'scene_n_parts', 'scene_n_copies']
    tmp = SUMMARY_FILE + '.partial'
    with open(tmp, 'w', newline='', encoding='utf-8', errors='surrogateescape') as f:
        w = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        w.writerow(front + list(fields))
        for key in order:
            aid, cid = key
            shp = os.path.join(CLASS_DIR, f"{cid}_members.shp") if cid else '(no footprint)'
            for r in sorted(groups[key], key=lambda r: (r.get('acq_center', ''), r.get('zip_path', ''))):
                site = format_sites(scene_sites(copies[dataset_key(r)])[0])
                w.writerow([aid, names.get(aid, ''), site, cid, shp, scenes[key],
                            r.get('n_parts') or '1', r.get('n_copies') or '1']
                           + [r.get(c, '') for c in fields])
    os.replace(tmp, SUMMARY_FILE)
    log(f"  5. {SUMMARY_FILE}: {sum(len(g) for g in groups.values())} scene(s), grouped by "
        f"area then class")


def write_centres(rows, row_class, row_area):
    """4. JAXA_scene_centres.shp: one point per scene that has a scene
    centre, including scenes without corners (HAS_CORNER = 0)."""
    feats = []
    for n, r in enumerate(rows, 1):
        if r.get('is_representative', 'yes') == 'no':
            continue
        lat, lon = to_float(r.get('center_lat')), to_float(r.get('center_lon'))
        if lat is None or lon is None:
            continue
        aid, aname = row_area.get(n, ('', ''))
        feats.append(((lon, lat), {
            'NAME': aname, 'AREA_ID': aid, 'CLASS_ID': row_class.get(n, ''), 'CSV_ROW': n,
            'HAS_CORNER': 1 if to_float(r.get('nw_lat')) is not None else 0,
            'ACQ_DATE': (r.get('acq_center') or r.get('acq_start') or '')[:10],
            'SAT': r.get('satellite', ''), 'MODE': r.get('observation_mode', ''),
            'SCENE_ID': r.get('scene_id', ''),
            'PROD_ID': r.get('product_id', ''), 'ORB_DIR': r.get('orbit_direction', ''),
            'POL': r.get('polarizations', ''), 'ACQ_CENTER': r.get('acq_center', ''),
            'INC_ANG': to_float(r.get('incidence_angle_center')),
            'CTR_SRC': r.get('center_source', ''), 'STATUS': r.get('status', ''),
            'ZIP': r.get('zip_path', '')}))
    write_shapefile(SHP_CENTRES, [('NAME', 'str'), ('AREA_ID', 'str'), ('CLASS_ID', 'str'),
                                  ('CSV_ROW', 'int'), ('HAS_CORNER', 'int'), ('ACQ_DATE', 'str'),
                                  ('SAT', 'str'), ('MODE', 'str'), ('SCENE_ID', 'str'),
                                  ('PROD_ID', 'str'), ('ORB_DIR', 'str'), ('POL', 'str'),
                                  ('ACQ_CENTER', 'str'), ('INC_ANG', 'real'), ('CTR_SRC', 'str'),
                                  ('STATUS', 'str'), ('ZIP', 'str')],
                    feats, points=True)
    n_only = sum(1 for _, a in feats if not a['HAS_CORNER'])
    log(f"  4. {SHP_CENTRES}: {len(feats)} scene centre point(s)"
        + (f", {n_only} of them from scenes without corners" if n_only else ""))


# ================================================================== QGIS
def open_qgis(files):
    files = [os.path.abspath(f) for f in files]
    exe = shutil.which('qgis') or shutil.which('qgis-ltr')
    if not exe and sys.platform == 'darwin':
        apps = (sorted(glob.glob('/Applications/QGIS*.app')) +
                sorted(glob.glob(os.path.expanduser('~/Applications/QGIS*.app'))))
        for app in apps:
            for binary in ('QGIS', 'qgis'):
                cand = os.path.join(app, 'Contents', 'MacOS', binary)
                if os.access(cand, os.X_OK):
                    exe = cand
                    break
            if exe:
                break
    if not exe:
        log("\n[WARN] QGIS not found (no `qgis` on PATH"
            + (", no /Applications/QGIS*.app" if sys.platform == 'darwin' else "")
            + "). Open these yourself:\n  " + "\n  ".join(files))
        return
    log(f"\nOpening QGIS: {exe}")
    subprocess.Popen([exe] + files, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)


# ================================================================== main
HELP_TEXT = (__doc__ or 'See the top of jaxa_catalogue.py').rstrip() + '\n'


class _HelpAction(argparse.Action):
    def __init__(self, option_strings, dest, **kw):
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS)

    def __call__(self, parser, namespace, values, option_string=None):
        print(HELP_TEXT)
        parser.exit()


def inspect(paths):
    """--inspect: full structural report on the named zips / parts / folders.
    Reads only; writes nothing."""
    import fnmatch
    if not paths:
        log("[ERROR] --inspect needs one or more zip files, parts or folders")
        sys.exit(1)
    files = []
    for p in paths:
        if os.path.isdir(p):
            for dirpath, dirnames, filenames in os.walk(p):
                dirnames.sort()
                files += [os.path.join(dirpath, n) for n in sorted(filenames)
                          if any(fnmatch.fnmatch(n, pat) for pat in FIND_PATTERNS)]
        elif os.path.isfile(p):
            files.append(p)
        else:
            log(f"[WARN] not found: {p}")
    entries = group_parts([os.path.abspath(f) for f in files])
    if not entries:
        log("[ERROR] no ALOS zip files or parts found in what was given")
        sys.exit(1)
    for entry in entries:
        parts = [entry] if isinstance(entry, str) else entry
        log("=" * 78)
        log(f"{product_name(parts)}  ({len(parts)} part(s))")
        for line in diagnose(parts)[0]:
            log(f"  {line}")
        try:
            a = Archive(entry)
            names = [n for n, _ in a.members() if not n.endswith('/')]
            log(f"  opens as: {a.kind}" + (f"  [normal reading failed: {a.damage}]" if a.damage else ""))
            log(f"  {len(names)} file(s) readable:")
            for n, s in a.members():
                if not n.endswith('/'):
                    log(f"    {s:>15,}  {n}")
            a.close()
        except ArchiveError as e:
            log(f"  cannot be read: {e}")
    log("=" * 78)


def _gdal_array_test(*python_flags):
    """Import GDAL's NumPy part in a separate, silent Python process.
    Returns (ok, error text)."""
    r = subprocess.run([sys.executable] + list(python_flags) + ['-c', 'import osgeo.gdal_array'],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return r.returncode == 0, r.stderr.decode('utf-8', 'replace')


def _numpy_location():
    """(version, folder) of the NumPy this Python imports, or ('', '')."""
    r = subprocess.run([sys.executable, '-c',
                        'import numpy, os; print(numpy.__version__); '
                        'print(os.path.dirname(os.path.dirname(numpy.__file__)))'],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out = r.stdout.decode('utf-8', 'replace').split('\n')
    return (out[0].strip(), out[1].strip()) if r.returncode == 0 and len(out) > 1 else ('', '')


def _pip(*args):
    """Run pip for this Python; retries without --break-system-packages for
    older pip versions. Returns True on success."""
    base = [sys.executable, '-m', 'pip'] + list(args)
    for extra in (['--break-system-packages'], []):
        r = subprocess.run(base + extra, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode == 0:
            return True
        if b'no such option' not in r.stderr and b'unrecognized' not in r.stderr:
            break
    return False


def _move_aside_numpy(folder):
    """Rename the numpy package (and its metadata / bundled libraries) in
    `folder` so Python no longer finds it. Reversible: the originals keep
    their names with '.disabled_by_jaxa_catalogue' added."""
    moved = []
    for name in sorted(os.listdir(folder)):
        if re.match(r'^numpy(-[^/]+\.dist-info|-[^/]+\.egg-info|\.libs|\.dylibs)?$', name):
            src = os.path.join(folder, name)
            dst = src + '.disabled_by_jaxa_catalogue'
            if os.path.exists(dst):
                shutil.rmtree(dst, ignore_errors=True)
            os.rename(src, dst)
            moved.append(dst)
    return moved


def fix_gdal_numpy():
    """Make GDAL's 'A module that was compiled using NumPy 1.x cannot be run
    in NumPy 2...' errors go away. They appear when GDAL's Python bindings
    were built for NumPy 1.x but a NumPy 2 is found first. In order:
      1. If a NumPy in the user's own folder (pip --user) shadows the NumPy
         GDAL was built with, remove it (pip uninstall; if pip cannot,
         rename it aside, reversibly) and restart.
      2. Otherwise install numpy<2 for this Python from a ready-made wheel
         (pip --user, never a source build) and restart.
      3. If neither works, mark GDAL's NumPy part as unavailable; GDAL then
         skips it silently. This script never uses it, so results are the
         same and no error is printed."""
    if os.environ.get('JAXA_NO_GDAL_ARRAY') == '1':
        return
    ok, err = _gdal_array_test()
    if ok or "No module named 'osgeo'" in err:
        return                                          # fine, or no GDAL at all
    clash = ('_ARRAY_API' in err or 'NumPy 1.x' in err or 'NumPy 2' in err
             or 'numpy.core.multiarray failed to import' in err
             or 'numpy.core._multiarray_umath' in err)
    if clash and os.environ.get('JAXA_NUMPY_FIX_TRIED') != '1':
        os.environ['JAXA_NUMPY_FIX_TRIED'] = '1'
        version, folder = _numpy_location()
        import site
        user_site = os.path.realpath(site.getusersitepackages())
        in_user_site = bool(folder) and os.path.realpath(folder) == user_site
        log(f"GDAL's Python bindings were built for NumPy 1.x, but NumPy {version or '2'} "
            f"is installed{(' in ' + folder) if folder else ''}. Fixing ...")
        fixed = False
        if in_user_site and _gdal_array_test('-s')[0]:
            # 1. the user-folder NumPy hides the one GDAL needs: take it away
            log(f"  removing NumPy {version} from your user folder, so the NumPy GDAL was "
                f"built with is used")
            for _ in range(3):                         # in case of several copies
                if not _pip('uninstall', '-y', 'numpy'):
                    break
                v, f = _numpy_location()
                if not f or os.path.realpath(f) != user_site:
                    break
            if _numpy_location()[1] and os.path.realpath(_numpy_location()[1]) == user_site:
                moved = _move_aside_numpy(user_site)
                if moved:
                    log("  pip could not remove it, so it was renamed aside (undo by removing "
                        "'.disabled_by_jaxa_catalogue' from these names):")
                    for m in moved:
                        log(f"    {m}")
            fixed = _gdal_array_test()[0]
        if not fixed:
            # 2. install numpy<2 (ready-made wheels only; never builds from source)
            log("  installing numpy<2 for this Python (pip --user, wheel only) ...")
            if _pip('install', '--user', '--quiet', '--only-binary=:all:', 'numpy<2'):
                fixed = _gdal_array_test()[0]
            else:
                log("  no numpy<2 wheel for this Python (or no pip / network)")
        if fixed:
            log("  fixed; restarting\n")
            sys.stdout.flush()
            os.execv(sys.executable, [sys.executable] + sys.argv)
    # 3. last resort, and the guarantee: GDAL skips its NumPy part quietly
    if clash:
        log("GDAL's NumPy part cannot be loaded with this NumPy; skipping it (this script "
            "does not use it, so results are unaffected)")
    os.environ['JAXA_NO_GDAL_ARRAY'] = '1'
    sys.modules['osgeo.gdal_array'] = None


def main():
    p = argparse.ArgumentParser(prog='jaxa_catalogue.py', add_help=False,
                                usage='python3 jaxa_catalogue.py [--merge | --inspect] [--verify] [--include-alos1] [--no-qgis] [PATH ...]   (-h for help)')
    p.add_argument('-h', '--help', action=_HelpAction)
    p.add_argument('--no-qgis', action='store_true')
    p.add_argument('--merge', action='store_true')
    p.add_argument('--inspect', action='store_true')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--include-alos1', action='store_true')
    p.add_argument('paths', nargs='*')
    args = p.parse_args()
    fix_gdal_numpy()
    global INCLUDE_ALOS1
    INCLUDE_ALOS1 = args.include_alos1

    if args.inspect:
        inspect(args.paths)
        return

    if args.merge:
        if os.path.exists(CSV_FILE):
            log(f"[ERROR] --merge writes {CSV_FILE} here, but that file already exists "
                f"in {os.getcwd()}. Move or rename it (or merge from another folder).")
            sys.exit(1)
        if args.paths:
            log("[WARN] PATH arguments are ignored with --merge (it searches the "
                "present folder tree for catalogue CSVs)")
        rows = drop_alos1(stage_merge(), 'from the merged catalogues')
        if not rows:
            sys.exit(1)
    elif os.path.exists(CSV_FILE):
        log(f"{CSV_FILE} exists: skipping the search and metadata stage "
            f"(rename or move it to repeat that stage)")
        if args.paths:
            log("[WARN] PATH arguments are ignored because the CSV already exists")
        rows = None
        if args.verify:          # verify the catalogue's copies and update it
            with open(CSV_FILE, newline='', encoding='utf-8', errors='surrogateescape') as f:
                rows = [{k: v for k, v in r.items()
                         if isinstance(k, str) and isinstance(v, str) and v != ''}
                        for r in csv.DictReader(f)]
            log(f"--verify: checking the copies listed in {CSV_FILE} and updating it")
            rows = drop_alos1(rows, f'from {CSV_FILE}')
        else:
            with open(CSV_FILE, newline='', encoding='utf-8', errors='surrogateescape') as f:
                n1 = sum(1 for r in csv.DictReader(f) if is_alos1_row(r))
            if n1 and not INCLUDE_ALOS1:
                log(f"[WARN] {CSV_FILE} lists {n1} ALOS-1 product(s) from an earlier run; "
                    f"they stay in the outputs until the catalogue is remade "
                    f"(rename the CSV and run again)")
    else:
        entries = stage_scan(args.paths)
        if not entries:
            log("[ERROR] no ALOS zip files found; nothing to catalogue")
            sys.exit(1)
        rows = drop_alos1(stage_metadata(entries), 'after reading their metadata')
        if not rows:
            log("[ERROR] nothing left to catalogue")
            sys.exit(1)

    if rows is not None:
        choose_representatives(rows, verify=args.verify)
        sort_rows(rows)
        write_csv(rows, CSV_FILE)
        report_sizes(rows)
        report_status(rows)
        report_scenes(rows)

    shapefiles, row_class, row_area = stage_shapefiles(CSV_FILE)
    if args.merge:
        write_summary(CSV_FILE, row_class, row_area)
    if shapefiles and not args.no_qgis:
        open_qgis(shapefiles)


if __name__ == '__main__':
    main()
