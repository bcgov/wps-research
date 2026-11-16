#!/usr/bin/env python3
"""
alos4_extract_zip.py

Extract SAR downloads into date-named folders (20yymmdd, i.e. yyyymmdd) next
to the original folders. Two kinds of download are recognised, one dataset
per folder:

ALOS-4 and ALOS-2 (JAXA): multi-part zips (*.zip.0001, *.zip.0002, ...) or
single zips, one subfolder per dataset or several datasets loose in one folder:
    ./0000626880/0000021542_0000626880_ALOS41712610250622HBQ-RD0606_1.1__-.zip.0001
    ./0000626880/0000021542_0000626880_ALOS41712610250622HBQ-RD0606_1.1__-.zip.0002
    ./0000021781_0000659468_ALOS2028271020-141201_FBDR1.1__A.zip
    ./0000487958_001001_ALOS2179381020-170918.zip
  Result: flat, every file directly in the date folder, whatever folder levels
  it had inside the zip:
    ./20250622/IMG-HH-ALOS41712610250622HBQ-RD0606-1.1__-
    ./20250622/ALOS41712610250622HBQ-RD0606_1.1__-.kml
    ...

SAOCOM (CONAE): one zip plus its .xemt metadata file
    ./EOL1ASARSAO1B15086383/S1B_OPER_SAR_EOSSP__CORE_L1A_OLF_20260618T120211.zip
    ./EOL1ASARSAO1B15086383/S1B_OPER_SAR_EOSSP__CORE_L1A_OLF_20260618T120211.xemt
  Result: the .xemt, and the zip's contents (Data/, Calibration/, ...) in a
  folder named after the product, under the ACQUISITION date (from the .xemt;
  here 2026-06-16, although the file name says 20260618):
    ./20260616/S1B_OPER_SAR_EOSSP__CORE_L1A_OLF_20260618T120211.xemt
    ./20260616/S1B_OPER_SAR_EOSSP__CORE_L1A_OLF_20260618T120211/Data/...
  SAOCOM is NOT flattened: SNAP opens a SAOCOM product through its .xemt file
  and then expects the Data/ and Calibration/ folders next to it, or inside a
  folder named after the product. The second form is used, so that several
  SAOCOM acquisitions on the same date can share a date folder without mixing.
  In gpt:  -Ssource=20260616/S1B_OPER_SAR_EOSSP__CORE_L1A_OLF_20260618T120211.xemt

Dates are ACQUISITION dates, read from each product's own metadata:
  ALOS-2/4: the summary*.txt file inside the zip (scene start time; if that
          file or its time field is missing, the date in the scene ID is used
          and a warning says so);
  SAOCOM: the .xemt file (acquisitionTime/startTime). The date in a SAOCOM
          file name is NOT the acquisition date (it is about when the product
          was made, often days later), so a SAOCOM product without a readable
          .xemt is not extracted.
Two products in one date folder with the same acquisition start time (the
same scene processed twice) are reported as a warning; both are extracted.

Datasets from the same date go into the same date folder, after checking that
no two files would get the same name there (a clash is reported as an error
and that date is not extracted).

The original folders and files are never modified. Multi-part zips are read
in order as one continuous stream (no concatenated copy is written to disk).

Progress:
    Workers send byte counts to the main process as they extract. The main
    process prints one aggregated status line (elapsed time, percent, GB,
    rate, ETA = time remaining) every few seconds, using a print lock so lines from
    different sources never interleave.

Resuming:
    It is safe to stop this script at any time (Ctrl-C, kill, power loss).
    Just run it again with the same arguments and it carries on:
      - each date folder is extracted into a hidden folder ".20yymmdd.partial"
        which is renamed to "20yymmdd" only when everything for it is done;
      - each file inside is written as "<name>.extracting" and renamed to
        its real name only after it is complete and its CRC checksum passed;
      - on restart, files already present with the correct size are kept,
        leftover "*.extracting" files are deleted and redone, and finished
        "20yymmdd" folders are skipped.

Usage:
    python3 alos4_extract_zip.py                 # current directory, 4 workers
    python3 alos4_extract_zip.py some_dir        # given (relative) directory
    python3 alos4_extract_zip.py some_dir -w 8   # 8 workers
    python3 alos4_extract_zip.py -i 10           # status every 10 seconds

No regular expressions are used: dates are taken by fixed position.
"""

import os
import io
import sys
import time
import queue
import shutil
import signal
import bisect
import heapq
import datetime
import argparse
import threading
import zipfile
import xml.etree.ElementTree as ET
import multiprocessing


# ---------------------------------------------------------------------------
# Fixed positions in the filenames
#
# ALOS-4:
#   0000021542_0000626880_ALOS41712610250622HBQ-RD0606_1.1__-.zip.0001
#   field[0]  field[1]   field[2] ...                  (split on '_')
#
#   field[2] = "ALOS41712610250622HBQ-RD0606"
#               0123456789012345678
#               ALOS4             -> [0:5]
#                       yymmdd    -> [12:18]  ("250622")
#
# ALOS-2:
#   0000021781_0000659468_ALOS2028271020-141201_FBDR1.1__A.zip
#   0000487958_001001_ALOS2179381020-170918.zip
#   field[0]  field[1] field[2] ...                       (split on '_')
#
#   field[2] = "ALOS2028271020-141201"   (anything from the first '.' on,
#               012345678901234567890     e.g. ".zip", is ignored)
#               ALOS2                -> [0:5]
#                             -      -> [14]
#                              yymmdd -> [15:21]  ("141201")
#
# SAOCOM:
#   S1B_OPER_SAR_EOSSP__CORE_L1A_OLF_20260618T120211.zip
#   field: 0   1    2   3     4(empty) 5  6   7   8       (split on '_';
#                                                 '__' gives an empty field 4)
#   field[0] = "S1A" or "S1B" (satellite)
#   field[8] = "20260618T120211"
#               01234567 8 9..14
#               yyyymmdd T hhmmss
#   (field[7] is "OLF" or "OLVF"; the field count is the same for both)
# ---------------------------------------------------------------------------
SCENE_FIELD_INDEX = 2
# mission prefix -> (date start, date end, position of a required '-' or None, launch date, name)
ALOS_MISSIONS = {
    "ALOS4": (12, 18, None, datetime.date(2024, 7, 1), "ALOS-4"),    # ALOS-4 launched 1 July 2024
    "ALOS2": (15, 21, 14, datetime.date(2014, 5, 24), "ALOS-2"),     # ALOS-2 launched 24 May 2014
}

SAOCOM_SATELLITES = ("S1A", "S1B")
SAOCOM_TIME_FIELD = 8
SAOCOM_NFIELDS = 9
SAOCOM_DATA_FOLDER = "Data"          # SNAP's SAOCOM reader needs this folder

TMP_SUFFIX = ".extracting"          # partially written file inside a dataset
CHUNK = 8 * 1024 * 1024             # read/write block size
REPORT_EVERY = 32 * 1024 * 1024     # worker sends progress after this many bytes


def check_date(d, label, launch, mission):
    if d > datetime.date.today():
        return "date string '%s' (%s) is in the future" % (label, d.isoformat())
    if d < launch:
        return "date string '%s' (%s) is before %s launch" % (label, d.isoformat(), mission)
    return None


def parse_date_string(filename):
    """ALOS-2/ALOS-4: return (date folder name yyyymmdd, mission prefix, error message)."""
    fields = filename.split("_")
    if len(fields) <= SCENE_FIELD_INDEX:
        return None, None, "filename has too few '_' separated fields"
    scene = fields[SCENE_FIELD_INDEX].split(".")[0]
    mission = scene[0:5]
    if mission not in ALOS_MISSIONS:
        return None, None, "field %d (%s) does not start with %s" % (
            SCENE_FIELD_INDEX, scene, " or ".join(sorted(ALOS_MISSIONS)))
    start, end, dash, launch, label = ALOS_MISSIONS[mission]
    if len(scene) < end:
        return None, mission, "scene field '%s' too short to hold a date" % scene
    if dash is not None and scene[dash] != "-":
        return None, mission, "scene field '%s' has no '-' at position %d" % (scene, dash)
    s = scene[start:end]
    if not s.isdigit():
        return None, mission, "date string '%s' is not all digits" % s
    yy = int(s[0:2])
    mm = int(s[2:4])
    dd = int(s[4:6])
    try:
        d = datetime.date(2000 + yy, mm, dd)
    except ValueError as e:
        return None, mission, "date string '%s' is not a valid yymmdd date (%s)" % (s, e)
    err = check_date(d, s, launch, label)
    if err:
        return None, mission, err
    return "20" + s, mission, None


def parse_saocom_name(filename):
    """SAOCOM: return (date folder name yyyymmdd, product name, error message)."""
    base = filename[:-len(".zip")] if filename.endswith(".zip") else filename
    fields = base.split("_")
    if len(fields) != SAOCOM_NFIELDS:
        return None, base, "expected %d '_' separated fields, found %d" % (SAOCOM_NFIELDS, len(fields))
    if fields[0] not in SAOCOM_SATELLITES:
        return None, base, "field 0 (%s) is not one of %s" % (fields[0], "/".join(SAOCOM_SATELLITES))
    t = fields[SAOCOM_TIME_FIELD]
    if len(t) != 15 or t[8] != "T" or not t[0:8].isdigit() or not t[9:15].isdigit():
        return None, base, "field %d (%s) is not a yyyymmddThhmmss time" % (SAOCOM_TIME_FIELD, t)
    s = t[0:8]
    try:
        d = datetime.date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
    except ValueError as e:
        return None, base, "date string '%s' is not a valid yyyymmdd date (%s)" % (s, e)
    hh, mi, ss = int(t[9:11]), int(t[11:13]), int(t[13:15])
    if hh > 23 or mi > 59 or ss > 60:
        return None, base, "time '%s' is not a valid hhmmss time" % t[9:15]
    err = check_date(d, s, datetime.date(2018, 10, 7), "SAOCOM-1A")  # SAOCOM-1A launched October 2018
    if err:
        return None, base, err
    return s, base, None


# ---------------------------------------------------------------------------
# Acquisition dates from the products' own metadata
# ---------------------------------------------------------------------------
ALOS_SUMMARY_KEYS = ("Img_SceneStartDateTime", "Img_SceneCenterDateTime", "Lbi_ObservationDate")


def date_from_timestamp(t):
    """'20260616 13:39:32...' or '2026-06-16T13:39:32...' -> ('20260616', date) or (None, None)."""
    t = t.strip().strip('"').strip("'")
    if len(t) >= 10 and t[4] == "-" and t[7] == "-":
        ymd = t[0:4] + t[5:7] + t[8:10]
    else:
        ymd = t[0:8]
    if len(ymd) != 8 or not ymd.isdigit():
        return None, None
    try:
        return ymd, datetime.date(int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]))
    except ValueError:
        return None, None


def alos_summary_time(text):
    """(key, value) of the first scene time found in an ALOS summary file
    (lines of the form key="value")."""
    values = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip().strip('"')
    for k in ALOS_SUMMARY_KEYS:
        if values.get(k):
            return k, values[k]
    return None, None


def xml_local(tag):
    return tag.split("}")[-1]


def saocom_acq_start(xml_bytes):
    """acquisitionTime/startTime from a SAOCOM .xemt (timeFrame/timePeriod/startTime
    as a fallback), or None."""
    root = ET.fromstring(xml_bytes)
    for parent_name in ("acquisitionTime", "timePeriod"):
        for el in root.iter():
            if xml_local(el.tag) == parent_name:
                for child in el:
                    if xml_local(child.tag) == "startTime" and child.text and child.text.strip():
                        return child.text.strip()
    return None


# ---------------------------------------------------------------------------
# Read several part-files as if they were one file (seekable, for zipfile)
# ---------------------------------------------------------------------------
class MultiPartFile(io.RawIOBase):
    def __init__(self, paths):
        self.paths = list(paths)
        self.sizes = [os.path.getsize(p) for p in self.paths]
        self.starts = []
        total = 0
        for sz in self.sizes:
            self.starts.append(total)
            total += sz
        self.total = total
        self.pos = 0
        self.cur_idx = -1
        self.cur_fh = None

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self.pos + offset
        elif whence == io.SEEK_END:
            new = self.total + offset
        else:
            raise ValueError("bad whence")
        if new < 0:
            raise ValueError("negative seek position")
        self.pos = new
        return self.pos

    def _open(self, idx):
        if idx != self.cur_idx:
            if self.cur_fh is not None:
                self.cur_fh.close()
            self.cur_fh = open(self.paths[idx], "rb")
            self.cur_idx = idx

    def readinto(self, b):
        mv = memoryview(b).cast("B")
        want = len(mv)
        got = 0
        while got < want and self.pos < self.total:
            idx = bisect.bisect_right(self.starts, self.pos) - 1
            offset = self.pos - self.starts[idx]
            n = min(want - got, self.sizes[idx] - offset)
            self._open(idx)
            self.cur_fh.seek(offset)
            chunk = self.cur_fh.readinto(mv[got:got + n])
            if not chunk:
                raise IOError("unexpected end of file in %s" % self.paths[idx])
            got += chunk
            self.pos += chunk
        return got

    def close(self):
        if self.cur_fh is not None:
            self.cur_fh.close()
            self.cur_fh = None
        super().close()


def open_zip(parts):
    raw = MultiPartFile(parts)
    buf = io.BufferedReader(raw, buffer_size=4 * 1024 * 1024)
    return zipfile.ZipFile(buf), buf


# ---------------------------------------------------------------------------
# Helpers shared by the main process and the workers
# ---------------------------------------------------------------------------
def path_parts(name):
    """Folder levels of a zip member path. Refuses absolute paths and '..'."""
    n = name.replace("\\", "/")
    if n.startswith("/"):
        raise ValueError("absolute path in zip: '%s'" % name)
    comps = []
    for c in n.split("/"):
        if c == "" or c == ".":
            continue
        if c == "..":
            raise ValueError("'..' in zip path: '%s'" % name)
        comps.append(c)
    if not comps:
        raise ValueError("empty file name in zip: '%s'" % name)
    return comps


def build_plan(ds, members):
    """Where each file of one dataset goes, relative to its date folder.
    Returns (entries, warnings); entries are (destination, zip member name or
    None, file to copy or None, size)."""
    entries = []
    warn = []
    files = [m for m in members if not m.is_dir()]
    if ds["kind"] in ALOS_MISSIONS:
        for m in files:
            entries.append((path_parts(m.filename)[-1], m.filename, None, m.file_size))
        return entries, warn
    # SAOCOM: keep the zip's structure, inside a folder named after the product
    prod = ds["product"]
    comps = [path_parts(m.filename) for m in files]
    tops = set(c[0] for c in comps)
    strip = (len(tops) == 1 and all(len(c) >= 2 for c in comps)
             and SAOCOM_DATA_FOLDER not in tops)          # zip has one top folder: drop it
    has_data = False
    for m, c in zip(files, comps):
        rel = c[1:] if strip else c
        if rel[0] == SAOCOM_DATA_FOLDER:
            has_data = True
        entries.append(("/".join([prod] + rel), m.filename, None, m.file_size))
    if not has_data:
        warn.append("%s: no %s/ folder in the zip; SNAP's SAOCOM reader expects one"
                    % (ds["src"], SAOCOM_DATA_FOLDER))
    if ds.get("xemt"):
        entries.append((prod + ".xemt", None, ds["xemt"], os.path.getsize(ds["xemt"])))
    elif not any(e[0].endswith(".xemt") for e in entries):
        warn.append("%s: no %s.xemt found; SNAP opens SAOCOM products through that file"
                    % (ds["src"], prod))
    return entries, warn


def check_clashes(plans):
    """Error message if two files (in one dataset, or in different datasets
    going to the same date folder) would get the same name."""
    seen = {}
    clashes = []
    for src, entries in plans:
        for dest, member, copy, size in entries:
            if dest.endswith(TMP_SUFFIX):
                # would be mistaken for a half-written file and deleted on resume
                return "file name ends with reserved suffix '%s': %s (%s)" % (TMP_SUFFIX, dest, src)
            if dest in seen:
                clashes.append("'%s' from %s and from %s" % (dest, seen[dest], src))
            else:
                seen[dest] = src
    if clashes:
        return "file name collision in the date folder: " + "; ".join(clashes[:10]) + (
            " (and %d more)" % (len(clashes) - 10) if len(clashes) > 10 else "")
    return None


def dest_path(root, dest):
    return os.path.join(root, *dest.split("/"))


def entry_done(tmp_dir, entry):
    """True if this file was already fully extracted in an earlier run."""
    p = dest_path(tmp_dir, entry[0])
    return os.path.isfile(p) and os.path.getsize(p) == entry[3]


def fmt_hms(seconds):
    seconds = int(max(0, seconds))
    return "%02d:%02d:%02d" % (seconds // 3600, (seconds // 60) % 60, seconds % 60)


def fmt_gb(n):
    return "%.2f GB" % (n / 1e9)


# ---------------------------------------------------------------------------
# Find jobs
# ---------------------------------------------------------------------------
def find_alos(folder):
    """ALOS datasets in one folder: ([(zip base name, [part paths])], [errors]).
    A dataset is either numbered parts (<name>.zip.0001, .0002, ...) or one
    <name>.zip whose scene field starts with ALOS2/ALOS4 (other single zips,
    e.g. SAOCOM's, are left alone)."""
    numbered = {}
    single = {}
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        if name.endswith(".zip"):
            fields = name.split("_")
            if len(fields) > SCENE_FIELD_INDEX and fields[SCENE_FIELD_INDEX][0:5] in ALOS_MISSIONS:
                single[name] = path
            continue
        dot = name.rfind(".")
        if dot < 0:
            continue
        base = name[:dot]
        suffix = name[dot + 1:]
        if not base.endswith(".zip") or not suffix.isdigit():
            continue
        numbered.setdefault(base, []).append((int(suffix), path))
    found = []
    errors = []
    for base in sorted(set(numbered) | set(single)):
        if base in numbered and base in single:
            errors.append("%s exists both as one zip and as numbered parts; not extracted" % base)
            continue
        if base in single:
            found.append((base, [single[base]]))
            continue
        parts = sorted(numbered[base])
        numbers = [p[0] for p in parts]
        expected = list(range(1, len(parts) + 1))
        if numbers != expected:
            errors.append("%s: part numbers %s are not consecutive starting at 1" % (base, numbers))
            continue
        found.append((base, [p[1] for p in parts]))
    return found, errors


def alos_dataset(src, base, parts, errors):
    """Dataset fields for one ALOS-2/ALOS-4 zip (or None, with the reason in errors).
    The date comes from the summary file inside the zip when it has one."""
    name_date, mission, err = parse_date_string(base)
    if err:
        errors.append("%s: %s" % (src, err))
        return None
    launch, label = ALOS_MISSIONS[mission][3], ALOS_MISSIONS[mission][4]
    ds = {"kind": mission, "src": src, "parts": parts, "date": name_date,
          "name_date": name_date, "acq": None,
          "date_src": "scene ID in the file name (no scene time found in a summary file)"}
    try:
        zf, buf = open_zip(parts)
        try:
            for m in zf.infolist():
                mbase = m.filename.replace("\\", "/").split("/")[-1]
                if (not m.is_dir() and mbase.lower().startswith("summary")
                        and mbase.lower().endswith(".txt") and m.file_size < 10 * 1024 * 1024):
                    key, val = alos_summary_time(zf.read(m).decode("utf-8", "replace"))
                    if key:
                        date, d2 = date_from_timestamp(val)
                        if date is None:
                            errors.append("%s: %s='%s' in %s is not a valid date" % (src, key, val, mbase))
                            return None
                        err = check_date(d2, date, launch, label)
                        if err:
                            errors.append("%s: acquisition %s" % (src, err))
                            return None
                        ds.update(date=date, acq=val, date_src="%s in %s" % (key, mbase))
                        break
        finally:
            zf.close()
            buf.close()
    except Exception as e:
        errors.append("%s: cannot read the zip: %s" % (src, e))
        return None
    return ds


def find_saocom(folder):
    """Return (dataset fields or None, error or None) for one folder."""
    zips = []
    for name in os.listdir(folder):
        if not name.endswith(".zip") or not os.path.isfile(os.path.join(folder, name)):
            continue
        if name.split("_")[0] in SAOCOM_SATELLITES:
            zips.append(name)
    if not zips:
        return None, None
    if len(zips) > 1:
        return None, "more than one SAOCOM zip in folder: %s" % ", ".join(sorted(zips))
    name_date, prod, err = parse_saocom_name(zips[0])
    if err:
        return None, err
    zpath = os.path.join(folder, zips[0])
    xemt = os.path.join(folder, prod + ".xemt")
    xml_bytes, where = None, None
    if os.path.isfile(xemt):
        with open(xemt, "rb") as f:
            xml_bytes, where = f.read(), prod + ".xemt"
    else:
        try:                                   # a copy inside the zip, if there is one
            zf, buf = open_zip([zpath])
            try:
                for m in zf.infolist():
                    if not m.is_dir() and m.filename.endswith(".xemt") and m.file_size < 50 * 1024 * 1024:
                        xml_bytes, where = zf.read(m), m.filename + " (inside the zip)"
                        break
            finally:
                zf.close()
                buf.close()
        except Exception as e:
            return None, "cannot read %s: %s" % (zips[0], e)
    if xml_bytes is None:
        return None, ("no %s.xemt: the acquisition date is unknown (the date in a SAOCOM file name "
                      "is not the acquisition date), so it is not extracted" % prod)
    try:
        start = saocom_acq_start(xml_bytes)
    except ET.ParseError as e:
        return None, "%s is not readable XML (%s)" % (where, e)
    if not start:
        return None, "no acquisition start time in %s; not extracted" % where
    date, d = date_from_timestamp(start)
    if date is None:
        return None, "acquisition start time '%s' in %s is not a valid date" % (start, where)
    err = check_date(d, date, datetime.date(2018, 10, 7), "SAOCOM-1A")
    if err:
        return None, "acquisition " + err
    return {"kind": "SAOCOM", "parts": [zpath], "date": date, "product": prod,
            "xemt": xemt if os.path.isfile(xemt) else None, "acq": start,
            "date_src": "acquisition time in " + where, "name_date": name_date}, None


def build_jobs(top):
    """Jobs, one per date folder: [(out_name, [datasets])], plus errors.
    Looks at zips directly in `top` and in each of its subfolders."""
    datasets = []
    errors = []
    # zips directly in the given folder (several datasets side by side)
    found, errs = find_alos(top)
    errors += errs
    for base, parts in found:
        ds = alos_dataset(base[:-len(".zip")], base, parts, errors)
        if ds:
            datasets.append(ds)
    # one folder per download
    for name in sorted(os.listdir(top)):
        folder = os.path.join(top, name)
        if not os.path.isdir(folder) or name.startswith("."):
            continue
        found, errs = find_alos(folder)
        errors += ["%s: %s" % (name, e) for e in errs]
        if found or errs:
            for base, parts in found:
                src = name if len(found) == 1 else name + "/" + base[:-len(".zip")]
                ds = alos_dataset(src, base, parts, errors)
                if ds:
                    datasets.append(ds)
            continue
        ds, err = find_saocom(folder)
        if err:
            errors.append("%s: %s" % (name, err))
            continue
        if ds:
            ds["src"] = name
            datasets.append(ds)
    jobs = {}
    for ds in datasets:
        jobs.setdefault(ds["date"], []).append(ds)
    return sorted(jobs.items()), errors


def date_notes(jobs):
    """Warnings: dates not taken from metadata, and the same acquisition delivered twice."""
    out = []
    for date, dsets in jobs:
        for ds in dsets:
            if ds.get("acq") is None:
                out.append("%s -> %s: date taken from the %s; not verified as the acquisition date"
                           % (ds["src"], date, ds["date_src"]))
        by_acq = {}
        for ds in dsets:
            if ds.get("acq"):
                by_acq.setdefault(ds["acq"], []).append(ds["src"])
        for acq, srcs in sorted(by_acq.items()):
            if len(srcs) > 1:
                out.append("%s: %s have the same acquisition start time (%s): the same scene "
                           "processed more than once; both are extracted, you probably need only one"
                           % (date, " and ".join(srcs), acq))
    return out


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
_progress_q = None
_parent_pid = None


def worker_init(q, parent_pid):
    global _progress_q, _parent_pid
    _progress_q = q
    _parent_pid = parent_pid
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # main process handles Ctrl-C
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def check_parent():
    # If the main process was killed outright, stop instead of running on
    # unattended. Everything written so far is resumable.
    if os.getppid() != _parent_pid:
        os._exit(1)


def extract_job(top, out_name, src_name, plans):
    """plans: [(dataset label, zip part paths, entries)].
    Returns (src_name, out_name, status, message). status: ok/skip/error."""
    t0 = time.time()
    final_dir = os.path.join(top, out_name)
    if os.path.exists(final_dir):
        return src_name, out_name, "skip", "already exists, left unchanged"
    tmp_dir = os.path.join(top, "." + out_name + ".partial")
    try:
        os.makedirs(tmp_dir, exist_ok=True)

        # remove half-written files from an interrupted earlier run
        for dirpath, dirnames, filenames in os.walk(tmp_dir):
            for fn in filenames:
                if fn.endswith(TMP_SUFFIX):
                    os.remove(os.path.join(dirpath, fn))

        _progress_q.put(("start", out_name))
        new_bytes = 0
        pending = 0
        nfiles = 0
        for label, parts, entries in plans:
            zf, buf = open_zip(parts)
            try:
                for entry in entries:
                    dest_rel, member, copy_from, size = entry
                    nfiles += 1
                    check_parent()
                    if entry_done(tmp_dir, entry):
                        continue
                    dest = dest_path(tmp_dir, dest_rel)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    tmp = dest + TMP_SUFFIX
                    src = zf.open(member) if member is not None else open(copy_from, "rb")
                    with src, open(tmp, "wb") as dst:
                        while True:
                            block = src.read(CHUNK)   # zip: CRC is checked at end of member
                            if not block:
                                break
                            dst.write(block)
                            pending += len(block)
                            if pending >= REPORT_EVERY:
                                _progress_q.put(("bytes", out_name, pending))
                                new_bytes += pending
                                pending = 0
                                check_parent()
                    os.replace(tmp, dest)
            finally:
                zf.close()
                buf.close()
        if pending:
            _progress_q.put(("bytes", out_name, pending))
            new_bytes += pending

        os.rename(tmp_dir, final_dir)
        return src_name, out_name, "ok", "%d files, %s written this run, %s" % (
            nfiles, fmt_gb(new_bytes), fmt_hms(time.time() - t0))
    except Exception as e:
        # keep tmp_dir: completed files in it are reused on the next run
        return src_name, out_name, "error", "%s: %s (re-run to resume)" % (
            type(e).__name__, e)


# ---------------------------------------------------------------------------
# Main process
# ---------------------------------------------------------------------------
class Interrupted(Exception):
    pass


def on_sigterm(signum, frame):
    raise Interrupted()


def main():
    ap = argparse.ArgumentParser(description="Extract ALOS-2/ALOS-4 (zip or multi-part zip) and SAOCOM (zip + .xemt) "
                                             "downloads into 20yymmdd folders.")
    ap.add_argument("directory", nargs="?", default=".",
                    help="parent directory holding the download folders (default: current)")
    ap.add_argument("-w", "--workers", type=int, default=4,
                    help="number of parallel workers (default: 4)")
    ap.add_argument("-i", "--interval", type=float, default=5.0,
                    help="seconds between status reports (default: 5)")
    args = ap.parse_args()

    top = os.path.normpath(args.directory)  # kept relative; never made absolute
    jobs, errors = build_jobs(top)
    print_lock = threading.Lock()

    def say(text):
        with print_lock:
            print(text)
            sys.stdout.flush()

    # --- pre-scan: what goes where (from each zip's table of contents),
    #     file name clashes, sizes, and what is already done from an earlier run
    total = {}       # out_name -> bytes for this date folder
    done = {}        # out_name -> bytes already extracted
    plans_of = {}    # out_name -> [(label, parts, entries)]
    label_of = {}    # out_name -> source folder names
    warns = date_notes(jobs)
    todo = []
    skipped = []
    for out_name, dsets in jobs:
        label = ", ".join(ds["src"] for ds in dsets)
        if os.path.exists(os.path.join(top, out_name)):
            skipped.append((label, out_name))
            continue
        try:
            plans = []
            for ds in dsets:
                zf, buf = open_zip(ds["parts"])
                try:
                    entries, w = build_plan(ds, zf.infolist())
                finally:
                    zf.close()
                    buf.close()
                warns += w
                plans.append((ds["src"], ds["parts"], entries))
            err = check_clashes([(p[0], p[2]) for p in plans])
            if err:
                raise ValueError(err)
        except Exception as e:
            errors.append("%s -> %s: not extracted: %s: %s" % (label, out_name, type(e).__name__, e))
            continue
        tmp_dir = os.path.join(top, "." + out_name + ".partial")
        total[out_name] = sum(e[3] for p in plans for e in p[2])
        done[out_name] = 0
        if os.path.isdir(tmp_dir):
            done[out_name] = sum(e[3] for p in plans for e in p[2] if entry_done(tmp_dir, e))
        plans_of[out_name] = plans
        label_of[out_name] = label
        todo.append((out_name, label, dsets))

    grand_total = sum(total[j[0]] for j in todo)
    resumed = sum(done[j[0]] for j in todo)

    say("Directory: %s" % top)
    say("Workers: %d   Status every %g s" % (args.workers, args.interval))
    for src_name, out_name in skipped:
        say("  %s -> %s  already finished, skipped" % (src_name, out_name))
    for out_name, label, dsets in todo:
        extra = ""
        if done[out_name]:
            extra = "  (resuming, %.1f%% already done)" % (
                100.0 * done[out_name] / max(1, total[out_name]))
        for k, ds in enumerate(dsets):
            what = ("%s, %s" % (ALOS_MISSIONS[ds["kind"]][4],
                                "1 zip" if len(ds["parts"]) == 1 else "%d parts" % len(ds["parts"]))
                    if ds["kind"] in ALOS_MISSIONS
                    else "SAOCOM, 1 zip%s" % (" + .xemt" if ds.get("xemt") else ""))
            moved = ""
            if ds.get("name_date") and ds["name_date"] != out_name:
                moved = "  (file name says %s; acquired %s)" % (ds["name_date"], out_name)
            say("  %s -> %s  %s%s%s" % (ds["src"], out_name, what, moved,
                                       ("  [%s total for %s%s]" % (fmt_gb(total[out_name]), out_name, extra))
                                       if k == len(dsets) - 1 else ""))
    say("Date folders to process: %d (%d datasets)   Total: %s%s" % (
        len(todo), sum(len(t[2]) for t in todo), fmt_gb(grand_total),
        ("   (already done: %s)" % fmt_gb(resumed)) if resumed else ""))
    say("")

    # --- run
    q = multiprocessing.Queue()
    state_lock = threading.Lock()
    active = set()          # datasets a worker is extracting right now
    ended = set()           # datasets that finished (ok or error)
    finished = [0]
    remaining = [len(todo)]
    t_start = time.time()
    session_bytes = [0]
    samples = []            # (time, session_bytes, number active) for the rate
    RATE_WINDOW = 30.0      # seconds of recent history used for the rate
    last_line = [None]      # last status line printed

    def on_result(res):
        src_name, out_name, status, msg = res
        with state_lock:
            finished[0] += 1
            remaining[0] -= 1
            active.discard(out_name)
            ended.add(out_name)
            if status == "ok":
                done[out_name] = total[out_name]
            if status == "error":
                errors.append("%s -> %s: %s" % (src_name, out_name, msg))
        if status == "error":
            say("ERROR %s -> %s: %s" % (src_name, out_name, msg))

    def on_error(exc):
        with state_lock:
            finished[0] += 1
            remaining[0] -= 1
            errors.append("worker crashed: %s" % exc)
        say("worker crashed: %s" % exc)

    def handle(msg):
        if msg[0] == "start":
            active.add(msg[1])
        elif msg[0] == "bytes":
            # cap at the dataset size: a finished dataset is already counted
            # as complete, so late-arriving messages must not count twice
            add = min(msg[2], total[msg[1]] - done[msg[1]])
            done[msg[1]] += add
            session_bytes[0] += add

    def take_sample():
        # called with state_lock held
        now = time.time()
        samples.append((now, session_bytes[0], len(active)))
        while len(samples) > 2 and samples[1][0] < now - RATE_WINDOW:
            samples.pop(0)

    def estimate_eta(rate, avg_active):
        """Seconds left, simulating the work queue: each worker extracts at
        (recent rate / workers that were busy), and a worker that finishes
        takes the next waiting dataset. This accounts for the slow tail at
        the end, when fewer datasets than workers are left."""
        per_worker = rate / max(1.0, avg_active)
        if per_worker <= 0:
            return None
        n_workers = max(1, args.workers)
        slots = []
        for out_name in active:
            slots.append(max(0, total[out_name] - done[out_name]) / per_worker)
        while len(slots) < n_workers:
            slots.append(0.0)                       # idle worker
        heapq.heapify(slots)
        for out_name, src_name, dsets in todo:     # waiting, in queue order
            if out_name in active or out_name in ended:
                continue
            t = heapq.heappop(slots)
            heapq.heappush(slots, t + max(0, total[out_name] - done[out_name]) / per_worker)
        return max(slots)

    def status_report():
        with state_lock:
            now = time.time()
            elapsed = now - t_start
            all_done = sum(done[j[0]] for j in todo)
            pct = 100.0 * all_done / max(1, grand_total)
            take_sample()
            t0, b0, a0 = samples[0]
            if now - t0 >= 1.0:
                rate = (session_bytes[0] - b0) / (now - t0)
                avg_active = sum(s[2] for s in samples) / float(len(samples))
            else:
                rate = session_bytes[0] / elapsed if elapsed > 0 else 0.0
                avg_active = float(len(active))
            eta = estimate_eta(rate, avg_active) if all_done < grand_total else 0.0
            # ETA is a duration (how long until finished), not a clock time
            if eta is None:
                eta_txt = "ETA --:--:-- remaining"
            else:
                eta_txt = "ETA %s remaining" % fmt_hms(eta)
            line = "[%s] %5.1f%%  %s / %s  |  %.1f MB/s  |  %s  |  date folders %d/%d done" % (
                fmt_hms(elapsed), pct, fmt_gb(all_done), fmt_gb(grand_total),
                rate / 1e6, eta_txt, finished[0], len(todo))
            if line == last_line[0]:
                return              # don't print the same status line twice
            last_line[0] = line
        say(line)

    pool = None
    old_term = signal.getsignal(signal.SIGTERM)
    try:
        if todo:
            pool = multiprocessing.Pool(processes=max(1, args.workers),
                                        initializer=worker_init,
                                        initargs=(q, os.getpid()))
            signal.signal(signal.SIGTERM, on_sigterm)
            # one task per dataset; a free worker takes the next one immediately
            for out_name, src_name, dsets in todo:
                pool.apply_async(extract_job, (top, out_name, src_name, plans_of[out_name]),
                                 callback=on_result, error_callback=on_error)
            pool.close()
            last = time.time()
            last_sample = 0.0
            while True:
                with state_lock:
                    left_jobs = remaining[0]
                if left_jobs <= 0:
                    break
                try:
                    msg = q.get(timeout=0.5)
                    with state_lock:
                        handle(msg)
                except queue.Empty:
                    pass
                if time.time() - last_sample >= 0.5:
                    with state_lock:
                        take_sample()
                    last_sample = time.time()
                if time.time() - last >= args.interval:
                    status_report()
                    last = time.time()
            pool.join()
            while True:  # drain anything left
                try:
                    msg = q.get_nowait()
                except queue.Empty:
                    break
                with state_lock:
                    handle(msg)
            status_report()
    except (KeyboardInterrupt, Interrupted):
        # ignore a second Ctrl-C / kill while shutting down the workers
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if pool is not None:
            pool.terminate()
            pool.join()
        say("")
        say("INTERRUPTED. Nothing is lost: run the same command again to resume.")
        sys.exit(130)
    finally:
        signal.signal(signal.SIGTERM, old_term)

    say("")
    say("Total time: %s" % fmt_hms(time.time() - t_start))
    if warns:
        say("WARNINGS (%d):" % len(warns))
        for w in warns:
            say("  " + w)
    if errors:
        say("ERRORS (%d):" % len(errors))
        for e in errors:
            say("  " + e)
        sys.exit(1)
    else:
        say("SUCCESS")


if __name__ == "__main__":
    main()
