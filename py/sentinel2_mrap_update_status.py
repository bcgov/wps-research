#!/usr/bin/env python3
"""mrap_status.py -- non-invasive status/ETA report for a running refresh_mrap.sh

READ-ONLY BY CONSTRUCTION.  This script never writes, moves, deletes or
signals anything.  It only: lists directories, stats files, reads the tail
of log files, and shells out to read-only inspection commands (ps, ss, df,
lsof).  It is safe to run repeatedly while the pipeline is mid-flight.

RUNS FROM ANYWHERE
------------------
Every path this script touches is absolute.  It never uses the current
working directory for anything, so it behaves identically whether you run
it from /data/mrap_bc, from $HOME, or from /.  The directory it inspects
is /data/mrap_bc by default; override with --mrap-dir.  The report header
always states which directory is being inspected.

LOG IDENTIFICATION (collision-proof)
------------------------------------
Two things could make the monitor read the wrong file: another tool
dropping a dated log into the same directory, or a stale log from an
unrelated job.  Both are handled:

  * PREFERRED naming -- distinctly labelled, cannot collide:
        .log_mrap_update_<YYYYMMDD>_<HHMMSS>.txt
        .log_fire_mapping_build_and_serve_<YYYYMMDD>_<HHMMSS>.txt
    See RECOMMENDED refresh_mrap.sh at the bottom of this docstring for
    the one-line change that produces the first of those.

  * LEGACY naming -- still supported, but content-verified:
        .log_<YYYYMMDD>_<HHMMSS>.txt
    Because that pattern is generic enough for anything to collide with,
    a candidate is only accepted if its head contains a recognisable
    pipeline fingerprint (a sync_recent.py / sentinel2_* invocation, a
    [PLAN] / [CLEANUP] / [DETECT] line, or the tile-loop banner).  Files
    that fail the check are counted and listed as ignored, so a collision
    is visible rather than silent.

  Labelled logs always win: if any .log_mrap_update_* exists, legacy logs
  are not consulted at all.  --strict-logs refuses legacy names outright.

WHAT IT MONITORS
----------------
refresh_mrap.sh runs two programs in sequence, each with its own log:

  1. sentinel2_mrap_update.py
     which in turn runs, in order:
       a. sync_recent.py                            (download new L2A ZIPs)
       b. sentinel2_swir_cloudmask_refine_L2_alltiles.py
            -> sentinel2_swir_cloudmask_refine.py --mrap, once per L2_T*
               tile folder (ABCD Random-Forest cloud-mask refinement, then
               per-tile MRAP compositing)
       c. sentinel2_mrap_merge.py                   (province-wide mosaic:
            gdalwarp resample of each tile to /ram, gdalbuildvrt, gdalwarp
            to <yyyymmdd>_mrap.bin)
       d. clean, then symlink / raster_warp_all.py / tar of the new output

  2. fire_mapping_build_and_serve_stack.py
     (stops the web server on port 8765, cleans /ram, rewrites the
      RASTERS=(...) line in run_fire_viirs_web.sh, restarts the server)

USAGE
-----
  mrap_status.py                     # one-shot report, from any directory
  mrap_status.py --watch             # refresh every 30s until Ctrl-C
  mrap_status.py --watch -n 10       # refresh every 10s
  mrap_status.py --sample 5          # sample file growth over 5s (default 3)
  mrap_status.py --no-sample         # skip growth sampling (instant report)
  mrap_status.py --no-lsof           # skip the /ram lsof scan
  mrap_status.py --tail 40           # show last 40 log lines (default 12)
  mrap_status.py --mrap-dir /other   # inspect a different archive
  mrap_status.py --strict-logs       # ignore legacy .log_<stamp>.txt entirely

RECOMMENDED refresh_mrap.sh
---------------------------
Only the log_file line changes; everything else is as you have it.  The
monitor reads both old and new names, so this can be adopted whenever
convenient -- no flag day required.

    #!/usr/bin/bash
    export PATH=/usr/local/bin:/home/ash/GitHub/wps-research/cpp:/home/ash/GitHub/bin/bin:$PATH
    cd /data/mrap_bc/
    timestamp=$(date +"%Y%m%d_%H%M%S")

    # Distinctly labelled so nothing else can collide with it:
    log_file=".log_mrap_update_$timestamp.txt"
    sentinel2_mrap_update.py >> "$log_file" 2>&1

    # stack the new data and reboot the server!
    fire_mapping_build_and_serve_stack.py \\
        >> ".log_fire_mapping_build_and_serve_$(date +%Y%m%d_%H%M%S).txt" 2>&1
"""

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# --- absolute defaults: nothing here depends on the current directory ------
DEFAULT_MRAP_DIR = Path("/data/mrap_bc")
DEFAULT_RAM_DIR = Path("/ram")
SERVER_PORT = 8765

MRAP_DIR = DEFAULT_MRAP_DIR      # replaced in main() from --mrap-dir
RAM_DIR = DEFAULT_RAM_DIR        # replaced in main() from --ram-dir

# --- log filename patterns -------------------------------------------------
# Preferred, distinctly labelled: .log_mrap_update_20260823_100001.txt
LABELLED_UPDATE_LOG_RE = re.compile(
    r"^\.log_mrap_update_(\d{8})_(\d{6})\.txt$")
# Legacy, generic (content-verified before use): .log_20260823_100001.txt
LEGACY_UPDATE_LOG_RE = re.compile(
    r"^\.log_(\d{8})_(\d{6})\.txt$")
# Second stage, already distinctly labelled:
STACK_LOG_RE = re.compile(
    r"^\.log_fire_mapping_build_and_serve_(\d{8})_(\d{6})\.txt$")
# Output mosaics: 20260823_mrap.bin
MRAP_OUT_RE = re.compile(r"^(\d{8})_mrap\.bin$")

# A legacy-named log must show at least one of these in its head to be
# accepted as ours.  Anything else in the directory is ignored and reported.
PIPELINE_FINGERPRINTS = [
    re.compile(r"sync_recent\.py"),
    re.compile(r"sentinel2_(mrap|swir)"),
    re.compile(r"run\((sync_recent|clean|sentinel2_)"),
    re.compile(r"\[CLEANUP\]|\[DETECT\]|\[PLAN\]|\[STATUS\]|\[MERGE done\]"),
    re.compile(r"Found\s+\d+\s+tile directories"),
    re.compile(r"\[AUTO\] Detected L2_\* folders|Level mode: L2"),
]

# Stage fingerprints: (stage key, human label, regex matched against log text)
STAGE_MARKERS = [
    ("sync",   "sync_recent.py (download new L2A products)",
     re.compile(r"run\(sync_recent\.py\)|sync_recent\.py")),
    ("refine", "sentinel2_swir_cloudmask_refine_L2_alltiles.py "
               "(per-tile cloud mask + MRAP)",
     re.compile(r"sentinel2_swir_cloudmask_refine_L2_alltiles\.py|"
                r"tile directories\.")),
    ("merge",  "sentinel2_mrap_merge.py (province-wide mosaic)",
     re.compile(r"sentinel2_mrap_merge\.py|"
                r"\[CLEANUP\] removing stale intermediates")),
    ("clean",  "clean / post-merge packaging (symlink, raster_warp_all.py, tar)",
     re.compile(r"run\(clean\)|raster_warp_all\.py|tar cvfz")),
]

# "[3/59] Processing L2_T10UFA ..."
TILE_PROGRESS_RE = re.compile(r"\[(\d+)/(\d+)\]\s+Processing\s+(\S+)")
# "Found 59 tile directories."
TILE_TOTAL_RE = re.compile(r"Found\s+(\d+)\s+tile directories")
# "[PLAN] 4 output file(s) to generate | N=1 | threads=1:"
PLAN_RE = re.compile(r"\[PLAN\]\s+(\d+)\s+output file\(s\) to generate")
# "[STATUS] 2/4 jobs done | avg_resample=41.2s | avg_merge=903.1s | ETA=00:31:04"
STATUS_RE = re.compile(
    r"\[STATUS\]\s+(\d+)/(\d+)\s+jobs done"
    r"(?:.*?avg_resample=([\d.]+)s)?"
    r"(?:.*?avg_merge=([\d.]+)s)?"
    r"(?:.*?ETA=([\d:]+|done))?"
)
# "[MERGE done]    target=20260822_mrap.bin | elapsed=903.1s"
MERGE_DONE_RE = re.compile(r"\[MERGE done\]\s+target=(\S+)")
# "SKIPPING (exists) 20260821_mrap.bin"
SKIP_RE = re.compile(r"^SKIPPING\s+\((.+?)\)\s+(\S+)")
# "...100 - done in 00:00:42."
DONE_IN_RE = re.compile(r"done in (\d{2}):(\d{2}):(\d{2})")
# GDAL's /ram-full signature
GDAL_ERR_RE = re.compile(r"ERROR 3:|I/O error|Insufficient disk space")

BAR_W = 34


# ---------------------------------------------------------------- utilities

def sh(cmd, timeout=15):
    """Run a read-only shell command, return stdout ('' on any failure).

    cwd is pinned to / so the command can never pick anything up from
    wherever the user happened to launch this script.
    """
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True, timeout=timeout, cwd="/")
        return r.stdout
    except Exception:
        return ""


def fmt_dur(seconds):
    if seconds is None:
        return "--:--:--"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_bytes(n):
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PiB"


def fmt_rate(bps):
    if bps is None or bps <= 0:
        return "stalled / idle"
    return fmt_bytes(bps) + "/s"


def bar(frac, width=BAR_W):
    frac = 0.0 if frac is None else max(0.0, min(1.0, frac))
    fill = int(round(frac * width))
    return "[" + "#" * fill + "." * (width - fill) + f"] {frac*100:5.1f}%"


def hr(title=""):
    if title:
        pad = max(0, 72 - len(title) - 3)
        return f"\n== {title} " + "=" * pad
    return "=" * 74


def age_str(path):
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


def tail_lines(path, n=200, max_bytes=2_000_000):
    """Read the last n lines of a file without loading the whole thing."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                fh.readline()          # discard partial line
            data = fh.read()
    except OSError:
        return []
    return data.decode("utf-8", errors="replace").splitlines()[-n:]


def head_text(path, max_bytes=65536):
    """Read the first chunk of a file, for fingerprint checking."""
    try:
        with open(path, "rb") as fh:
            return fh.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def parse_log_stamp(m):
    """(date, time) regex groups -> datetime."""
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


# ------------------------------------------------------------ process table

def process_table():
    """All pipeline-related processes, newest-first, with elapsed seconds."""
    out = sh("ps -eo pid,ppid,etimes,etime,rss,pcpu,args --no-headers")
    pats = [
        ("refresh_mrap.sh",                                "refresh_mrap.sh"),
        ("sentinel2_mrap_update.py",                       "update (orchestrator)"),
        ("sync_recent.py",                                 "sync_recent"),
        ("sentinel2_swir_cloudmask_refine_L2_alltiles.py", "refine (all tiles)"),
        ("sentinel2_swir_cloudmask_refine.py",             "refine (one tile)"),
        ("sentinel2_mrap_merge.py",                        "merge (mosaic)"),
        ("fire_mapping_build_and_serve_stack.py",          "stack + serve"),
        ("gdalwarp",                                       "gdalwarp"),
        ("gdalbuildvrt",                                   "gdalbuildvrt"),
        ("sentinel2_mrap_QA",                              "QA"),
        ("raster_warp_all.py",                             "raster_warp_all"),
        ("batch_fire_mapping_viirs_web",                   "web server"),
    ]
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        pid, ppid, etimes, etime, rss, pcpu, args_ = parts
        if "mrap_status.py" in args_ or "ps -eo" in args_:
            continue
        for needle, label in pats:
            if needle in args_:
                try:
                    et = int(etimes)
                except ValueError:
                    et = 0
                try:
                    rssk = int(rss)
                except ValueError:
                    rssk = 0
                rows.append(dict(pid=pid, ppid=ppid, etimes=et, etime=etime,
                                 rss_kb=rssk, pcpu=pcpu, label=label,
                                 args=args_.strip()))
                break
    rows.sort(key=lambda r: -r["etimes"])
    return rows


def has(rows, label):
    return any(r["label"] == label for r in rows)


# ----------------------------------------------------------------- log find

def looks_like_pipeline_log(path):
    """True if a legacy-named log really is one of ours."""
    head = head_text(path)
    if not head.strip():
        # Empty file: accept only if very fresh (a run that just started
        # and has not flushed its first line yet).
        a = age_str(path)
        return a is not None and a < 600
    return any(rx.search(head) for rx in PIPELINE_FINGERPRINTS)


def find_logs(strict=False):
    """Locate update and stack logs under MRAP_DIR.

    Returns (newest_update, newest_stack, durations, provenance) where
    provenance records how logs were identified and what was ignored.
    """
    labelled, legacy, stack, ignored = [], [], [], []
    prov = dict(source=None, labelled_count=0, legacy_count=0,
                ignored=[], dir=str(MRAP_DIR), error=None)

    try:
        entries = list(MRAP_DIR.iterdir())
    except OSError as exc:
        prov["error"] = str(exc)
        return None, None, [], prov

    for p in sorted(entries):
        if not p.is_file():
            continue
        name = p.name

        m = STACK_LOG_RE.match(name)
        if m:
            stack.append((parse_log_stamp(m), p))
            continue

        m = LABELLED_UPDATE_LOG_RE.match(name)
        if m:
            labelled.append((parse_log_stamp(m), p))
            continue

        m = LEGACY_UPDATE_LOG_RE.match(name)
        if m:
            if looks_like_pipeline_log(p):
                legacy.append((parse_log_stamp(m), p))
            else:
                ignored.append(name)
            continue

    prov["labelled_count"] = len(labelled)
    prov["legacy_count"] = len(legacy)
    prov["ignored"] = ignored

    # Labelled logs win outright.  Legacy only if no labelled log exists
    # and --strict-logs was not requested.
    if labelled:
        update = labelled
        prov["source"] = "labelled (.log_mrap_update_<stamp>.txt)"
        if legacy:
            prov["source"] += (f"; {len(legacy)} legacy log(s) present "
                               f"but not used")
    elif legacy and not strict:
        update = legacy
        prov["source"] = ("legacy (.log_<stamp>.txt), content-verified -- "
                          "consider switching to .log_mrap_update_<stamp>.txt")
    else:
        update = []
        prov["source"] = ("none found"
                          + (" (--strict-logs: legacy names ignored)"
                             if strict and legacy else ""))

    update.sort()
    stack.sort()

    # Historical wall-clock durations: mtime - start, for every run but the
    # newest (which may still be in flight).
    durations = []
    for started, p in update[:-1]:
        try:
            dur = p.stat().st_mtime - started.timestamp()
        except OSError:
            continue
        if 60 < dur < 86400 * 2:       # sanity window
            durations.append(dur)

    return (update[-1] if update else None,
            stack[-1] if stack else None,
            durations, prov)


# -------------------------------------------------------------- log parsing

def analyse_update_log(path):
    """Everything we can learn from the sentinel2_mrap_update.py log."""
    lines = tail_lines(path, n=4000)
    text = "\n".join(lines)
    info = dict(lines=lines, stage=None, stage_label=None,
                tile_i=None, tile_n=None, tile_name=None,
                plan_total=None, merge_done=[], skipped=[],
                status=None, gdal_errors=0, tile_done_times=[])

    last_pos, last_stage = -1, None
    for key, label, rx in STAGE_MARKERS:
        for m in rx.finditer(text):
            if m.start() > last_pos:
                last_pos, last_stage = m.start(), (key, label)
    if last_stage:
        info["stage"], info["stage_label"] = last_stage

    for line in lines:
        m = TILE_PROGRESS_RE.search(line)
        if m:
            info["tile_i"] = int(m.group(1))
            info["tile_n"] = int(m.group(2))
            info["tile_name"] = m.group(3)
        m = TILE_TOTAL_RE.search(line)
        if m and info["tile_n"] is None:
            info["tile_n"] = int(m.group(1))
        m = PLAN_RE.search(line)
        if m:
            info["plan_total"] = int(m.group(1))
            info["merge_done"] = []          # a new plan resets the count
        m = MERGE_DONE_RE.search(line)
        if m:
            info["merge_done"].append(m.group(1))
        m = SKIP_RE.search(line)
        if m:
            info["skipped"].append((m.group(1), m.group(2)))
        m = STATUS_RE.search(line)
        if m:
            info["status"] = dict(done=int(m.group(1)), total=int(m.group(2)),
                                  avg_resample=m.group(3),
                                  avg_merge=m.group(4), eta=m.group(5))
        m = DONE_IN_RE.search(line)
        if m:
            info["tile_done_times"].append(
                int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)))
        if GDAL_ERR_RE.search(line):
            info["gdal_errors"] += 1

    return info


# ----------------------------------------------------- filesystem sampling

def sample_growth(paths, seconds):
    """Sample sizes of paths twice, return {path: (size, bytes_per_sec)}."""
    first = {}
    for p in paths:
        try:
            first[p] = p.stat().st_size
        except OSError:
            first[p] = None
    if seconds <= 0:
        return {p: (s, None) for p, s in first.items()}
    time.sleep(seconds)
    out = {}
    for p in paths:
        try:
            now = p.stat().st_size
        except OSError:
            now = None
        rate = None
        if now is not None and first.get(p) is not None:
            rate = (now - first[p]) / seconds
        out[p] = (now, rate)
    return out


def outputs_state():
    """Completed <yyyymmdd>_mrap.bin outputs and the typical full size."""
    done, sizes = [], []
    try:
        entries = list(MRAP_DIR.iterdir())
    except OSError:
        return [], None
    for p in entries:
        m = MRAP_OUT_RE.match(p.name)
        if not m:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        done.append(dict(date=m.group(1), path=p, size=st.st_size,
                         mtime=st.st_mtime, age=time.time() - st.st_mtime))
        sizes.append(st.st_size)
    done.sort(key=lambda d: d["date"])
    typical = None
    if sizes:
        sizes.sort()
        typical = sizes[len(sizes) // 2]      # median = full-size reference
    return done, typical


def ram_intermediates():
    try:
        return sorted(RAM_DIR.glob("*_resample.bin"))
    except OSError:
        return []


def disk_line(mount):
    if not Path(mount).exists():
        return f"{mount}: not present on this host"
    out = sh(f"df -h {mount}")
    rows = out.strip().splitlines()
    return rows[-1] if len(rows) >= 2 else f"{mount}: df returned nothing"


def ram_held_by_deleted():
    """Space held by deleted-but-open files on /ram (the classic tmpfs trap)."""
    if not RAM_DIR.exists():
        return []
    out = sh(f"lsof +D {RAM_DIR} 2>/dev/null | grep '(deleted)'", timeout=25)
    if not out.strip():
        return []
    seen, rows = set(), []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        cmd, pid, size = parts[0], parts[1], parts[-4]
        name = parts[-2] if parts[-1] == "(deleted)" else parts[-1]
        key = (pid, name)
        if key in seen:
            continue
        seen.add(key)
        try:
            size = int(size)
        except ValueError:
            size = 0
        rows.append(dict(cmd=cmd, pid=pid, size=size, name=name))
    return rows


def server_up():
    return f":{SERVER_PORT}" in sh(f"ss -ltn '( sport = :{SERVER_PORT} )'")


# ------------------------------------------------------------------ report

def report(args):
    now = datetime.now()
    try:
        cwd = Path.cwd()
    except OSError:
        cwd = Path("(unavailable)")

    print(hr())
    print(f" MRAP PIPELINE STATUS  --  {now:%Y-%m-%d %H:%M:%S}   (read-only probe)")
    print(f" Inspecting    : {MRAP_DIR}")
    print(f" Scratch/ram   : {RAM_DIR}")
    print(f" Launched from : {cwd}   (irrelevant -- all paths are absolute)")
    print(hr())

    procs = process_table()
    newest_update, newest_stack, durations, prov = find_logs(args.strict_logs)

    # ---------------------------------------------------------- 1. running?
    print(hr("1. PROCESSES"))
    if not procs:
        print("  No pipeline processes found.")
    else:
        print(f"  {'PID':>8} {'ELAPSED':>10} {'RSS':>9} {'%CPU':>5}  WHAT")
        for r in procs[:18]:
            print(f"  {r['pid']:>8} {r['etime']:>10} "
                  f"{fmt_bytes(r['rss_kb']*1024):>9} {r['pcpu']:>5}  {r['label']}")
        if len(procs) > 18:
            print(f"  ... and {len(procs)-18} more")

    running_top = has(procs, "refresh_mrap.sh")
    running_update = has(procs, "update (orchestrator)")
    running_stack = has(procs, "stack + serve")

    # ------------------------------------------------------- 2. log sources
    print(hr("2. LOG IDENTIFICATION"))
    print(f"  Directory     : {prov['dir']}")
    print(f"  Using         : {prov['source']}")
    print(f"  Labelled logs : {prov['labelled_count']}  "
          f"(.log_mrap_update_<stamp>.txt)")
    print(f"  Legacy logs   : {prov['legacy_count']}  "
          f"(.log_<stamp>.txt, content-verified)")
    if prov["ignored"]:
        print(f"  IGNORED       : {len(prov['ignored'])} dated log(s) here that "
              f"are NOT pipeline logs:")
        for n in prov["ignored"][:6]:
            print(f"                  {n}")
        if len(prov["ignored"]) > 6:
            print(f"                  ... and {len(prov['ignored'])-6} more")
    else:
        print("  IGNORED       : none (no colliding dated logs present)")
    if prov.get("error"):
        print(f"  ERROR reading directory: {prov['error']}")

    # ------------------------------------------------------- 3. current run
    print(hr("3. CURRENT RUN"))
    if newest_update is None:
        print(f"  No pipeline log found in {MRAP_DIR} -- nothing to report.")
        print("  (If the run is very new, its log may not exist yet.)")
        return

    started, log_path = newest_update
    elapsed = (now - started).total_seconds()
    log_age = age_str(log_path)
    print(f"  Log        : {log_path}")
    print(f"  Started    : {started:%Y-%m-%d %H:%M:%S}   ({fmt_dur(elapsed)} ago)")
    try:
        print(f"  Log size   : {fmt_bytes(log_path.stat().st_size)}")
    except OSError:
        print("  Log size   : (unreadable)")
    print(f"  Last write : {fmt_dur(log_age)} ago", end="")
    if log_age is not None and log_age > 900 and (running_top or running_update):
        print("   <-- WARNING: log quiet for >15 min while processes are alive")
    elif log_age is not None and log_age > 900:
        print("   (run appears finished)")
    else:
        print()

    info = analyse_update_log(log_path)

    # ------------------------------------------------------------ 4. stage
    print(hr("4. STAGE"))
    stages = [k for k, _, _ in STAGE_MARKERS]
    labels = {k: lbl for k, lbl, _ in STAGE_MARKERS}
    cur = info["stage"]
    for k in stages:
        if cur is None:
            mark = " "
        elif k == cur:
            mark = ">"
        elif stages.index(k) < stages.index(cur):
            mark = "x"
        else:
            mark = " "
        print(f"   [{mark}] {labels[k]}")
    print()
    if cur is None:
        print("  Stage undetermined (log tail shows no stage marker yet).")
    else:
        print(f"  Current: {labels[cur]}")

    # --------------------------------------------- 5. within-stage progress
    print(hr("5. PROGRESS WITHIN STAGE"))

    if info["tile_i"] and info["tile_n"]:
        i, n = info["tile_i"], info["tile_n"]
        print("  Tile loop (sentinel2_swir_cloudmask_refine.py, "
              "one run per L2_T* folder):")
        print(f"    {bar(i/n)}   tile {i} of {n}   now: {info['tile_name']}")
        if cur == "refine" and elapsed > 0 and i > 0:
            per_tile = elapsed / i
            print(f"    ~{fmt_dur(per_tile)} per tile so far "
                  f"-> ~{fmt_dur(per_tile*(n-i))} left in this stage")
    else:
        print("  Tile loop  : no [i/N] Processing line seen yet.")

    if info["plan_total"] is not None:
        done = len(info["merge_done"])
        tot = info["plan_total"]
        print("\n  Merge plan (sentinel2_mrap_merge.py, one job per output date):")
        print(f"    {bar(done/tot if tot else 0)}   {done} of {tot} mosaics written")
        if info["merge_done"]:
            print(f"    completed: {', '.join(info['merge_done'][-6:])}")
    else:
        print("\n  Merge plan : no [PLAN] line seen yet.")

    st = info["status"]
    if st:
        bits = [f"{st['done']}/{st['total']} jobs"]
        if st["avg_resample"]:
            bits.append(f"avg resample {float(st['avg_resample']):.0f}s")
        if st["avg_merge"]:
            bits.append(f"avg merge {float(st['avg_merge']):.0f}s")
        print(f"    merge script's own [STATUS]: {' | '.join(bits)}")
        if st["eta"]:
            print(f"    merge script's own ETA    : {st['eta']}")

    if info["skipped"]:
        reasons = {}
        for why, _what in info["skipped"]:
            reasons[why] = reasons.get(why, 0) + 1
        print("\n  Dates skipped by the merge planner:")
        for why, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>4}  {why}")

    # ------------------------------------------------- 6. file-level motion
    print(hr("6. FILE-LEVEL ACTIVITY"))
    done_outputs, typical = outputs_state()

    in_progress = [d for d in done_outputs
                   if typical and d["size"] < typical * 0.995 and d["age"] < 7200]
    inters = ram_intermediates()
    watch = [d["path"] for d in in_progress] + inters[:6]

    sample_s = 0 if args.no_sample else args.sample_seconds
    if sample_s > 0 and watch:
        print(f"  Sampling file growth over {sample_s}s ...")
    growth = sample_growth(watch, sample_s)

    if in_progress:
        for d in in_progress:
            size, rate = growth.get(d["path"], (d["size"], None))
            frac = size / typical if (typical and size) else None
            print(f"\n  Mosaic being written: {d['path'].name}")
            print(f"    {bar(frac)}   {fmt_bytes(size)} of ~{fmt_bytes(typical)}")
            print(f"    write rate : {fmt_rate(rate)}")
            if rate and rate > 0 and typical and size:
                print(f"    this file  : ~{fmt_dur((typical-size)/rate)} remaining")
    else:
        print("  No partially-written <yyyymmdd>_mrap.bin detected at top level.")

    if inters:
        print(f"\n  {RAM_DIR} resample intermediates in flight: {len(inters)}")
        for p in inters[:6]:
            size, rate = growth.get(p, (None, None))
            print(f"    {p.name:<58} {fmt_bytes(size):>9}  {fmt_rate(rate)}")
        if len(inters) > 6:
            print(f"    ... and {len(inters)-6} more")
    else:
        print(f"\n  {RAM_DIR} resample intermediates in flight: none")

    if done_outputs:
        newest = done_outputs[-1]
        print(f"\n  Top-level mosaics present : {len(done_outputs)}")
        print(f"  Newest by date            : {newest['date']}_mrap.bin "
              f"({fmt_bytes(newest['size'])}, written {fmt_dur(newest['age'])} ago)")
        print(f"  Full-size reference       : {fmt_bytes(typical)} "
              f"(median of existing)")

    # ------------------------------------------------------------ 7. disks
    print(hr("7. DISK / RAMDISK"))
    print(f"  {disk_line(str(RAM_DIR))}")
    print(f"  {disk_line(str(MRAP_DIR))}")
    if not args.no_lsof:
        held = ram_held_by_deleted()
        if held:
            total = sum(h["size"] for h in held)
            print(f"\n  {RAM_DIR} space held by DELETED-but-open files: "
                  f"{fmt_bytes(total)}")
            agg = {}
            for h in held:
                k = (h["cmd"], h["pid"])
                agg[k] = agg.get(k, 0) + h["size"]
            for (cmd, pid), sz in sorted(agg.items(), key=lambda kv: -kv[1])[:8]:
                print(f"    {cmd:<16} pid {pid:<9} {fmt_bytes(sz):>10}")
            print("    (closing those processes returns the space immediately)")
        else:
            print(f"\n  No deleted-but-open files holding space on {RAM_DIR}.")

    if info["gdal_errors"]:
        print(f"\n  !! {info['gdal_errors']} GDAL write/IO error line(s) in this "
              f"log -- usually means {RAM_DIR} filled up mid-write.")
        print("     Re-check sections 6/7 and grep the log for 'ERROR 3'.")

    # ------------------------------------------------------- 8. second half
    print(hr("8. STAGE 2 (fire_mapping_build_and_serve_stack.py)"))
    if newest_stack:
        s_started, s_path = newest_stack
        s_age = age_str(s_path)
        fresh = s_started >= started
        print(f"  Log       : {s_path.name}")
        print(f"  Started   : {s_started:%Y-%m-%d %H:%M:%S} "
              f"({'this run' if fresh else 'PREVIOUS run -- not reached yet'})")
        print(f"  Last write: {fmt_dur(s_age)} ago")
        if fresh:
            for line in tail_lines(s_path, n=6):
                print(f"    | {line}")
    else:
        print("  No fire_mapping_build_and_serve log found yet.")
    print(f"  Web server on port {SERVER_PORT}: "
          f"{'UP' if server_up() else 'DOWN (expected while stage 2 runs)'}")

    # ------------------------------------------------------------- 9. ETA
    print(hr("9. OVERALL ESTIMATE"))
    if durations:
        ds = sorted(durations)
        med = ds[len(ds) // 2]
        print(f"  Historical runs measured   : {len(ds)}")
        print(f"  Typical wall-clock (median): {fmt_dur(med)} "
              f"(range {fmt_dur(ds[0])} to {fmt_dur(ds[-1])})")
        print(f"  Elapsed this run           : {fmt_dur(elapsed)}")
        print(f"  {bar(elapsed/med if med else None)}  vs typical run")
        if elapsed < med:
            eta = med - elapsed
            print(f"  Naive ETA (median-based)   : ~{fmt_dur(eta)} remaining "
                  f"-> done around {(now + timedelta(seconds=eta)):%H:%M:%S}")
        else:
            print(f"  Already {fmt_dur(elapsed-med)} past the median run length.")
            print("  (Longer than usual is normal when several dates are being "
                  "regenerated at once -- check section 5 for the real job count.)")
    else:
        print("  Not enough completed logs to estimate a typical run length.")

    if st and st.get("eta") and st["eta"] != "done":
        print(f"  Merge stage's own ETA      : {st['eta']} (most reliable once "
              f"merging has started)")

    verdict = ("RUNNING" if (running_top or running_update or running_stack)
               else "NOT RUNNING (no pipeline processes alive)")
    print(f"\n  Verdict: {verdict}")

    # -------------------------------------------------------- 10. log tail
    if args.tail > 0:
        print(hr(f"10. LAST {args.tail} LOG LINES"))
        for line in info["lines"][-args.tail:]:
            print(f"  | {line[:200]}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Non-invasive status and ETA report for a running "
                    "refresh_mrap.sh pipeline. Read-only: never writes, "
                    "moves, deletes or signals anything. Runs correctly "
                    "from any working directory.")
    ap.add_argument("--watch", action="store_true",
                    help="repeat the report until Ctrl-C")
    ap.add_argument("-n", "--interval", type=int, default=30,
                    help="seconds between refreshes with --watch (default 30)")
    ap.add_argument("--sample", dest="sample_seconds", type=int, default=3,
                    help="seconds to sample file growth for write rates "
                         "(default 3)")
    ap.add_argument("--no-sample", action="store_true",
                    help="skip the growth sample entirely (instant report)")
    ap.add_argument("--no-lsof", action="store_true",
                    help="skip the lsof scan of the ramdisk "
                         "(faster on a busy box)")
    ap.add_argument("--tail", type=int, default=12,
                    help="log lines to show at the end (default 12, 0 to omit)")
    ap.add_argument("--mrap-dir", "--dir", dest="mrap_dir", type=str,
                    default=str(DEFAULT_MRAP_DIR),
                    help=f"archive directory to inspect "
                         f"(default {DEFAULT_MRAP_DIR})")
    ap.add_argument("--ram-dir", type=str, default=str(DEFAULT_RAM_DIR),
                    help=f"ramdisk/scratch directory "
                         f"(default {DEFAULT_RAM_DIR})")
    ap.add_argument("--strict-logs", action="store_true",
                    help="only accept distinctly-labelled "
                         ".log_mrap_update_<stamp>.txt logs; ignore legacy "
                         ".log_<stamp>.txt entirely")
    args = ap.parse_args()

    global MRAP_DIR, RAM_DIR
    # expanduser + resolve makes these absolute regardless of launch directory
    MRAP_DIR = Path(args.mrap_dir).expanduser().resolve()
    RAM_DIR = Path(args.ram_dir).expanduser().resolve()

    if not MRAP_DIR.is_dir():
        print(f"ERROR: {MRAP_DIR} is not a directory", file=sys.stderr)
        sys.exit(2)

    if not args.watch:
        report(args)
        return

    try:
        while True:
            # ANSI clear: no dependency on `clear` being on PATH or on cwd
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            report(args)
            print(f"  (refreshing every {args.interval}s -- Ctrl-C to stop)")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()


