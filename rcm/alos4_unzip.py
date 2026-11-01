#!/usr/bin/env python3
"""
20261101 alos4_extract_zip.py

Extract ALOS-4 multi-part zip downloads (e.g. *.zip.0001, *.zip.0002) into
date-named folders (20yymmdd) next to the original folders.

Layout expected (run from, or point at, the parent directory):

    ./0000626880/0000021542_0000626880_ALOS41712610250622HBQ-RD0606_1.1__-.zip.0001
    ./0000626880/0000021542_0000626880_ALOS41712610250622HBQ-RD0606_1.1__-.zip.0002

Result (flat: every file directly in the date folder, whatever folder
levels it had inside the zip):

    ./20250622/IMG-HH-ALOS41712610250622HBQ-RD0606-1.1__-
    ./20250622/LED-ALOS41712610250622HBQ-RD0606-1.1__-
    ./20250622/ALOS41712610250622HBQ-RD0606_1.1__-.kml
    ...

The original folders and files are never modified. The parts are read in
order as one continuous stream (no concatenated copy is written to disk).

Progress:
    Workers send byte counts to the main process as they extract. The main
    process prints one aggregated status line (elapsed time, percent, GB,
    rate, ETA = time remaining) every few seconds, using a print lock so lines from
    different sources never interleave.

Resuming:
    It is safe to stop this script at any time (Ctrl-C, kill, power loss).
    Just run it again with the same arguments and it carries on:
      - each dataset is extracted into a hidden folder ".20yymmdd.partial"
        which is renamed to "20yymmdd" only when the whole dataset is done;
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

No regular expressions are used: the date is taken by fixed position.
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
import multiprocessing


# ---------------------------------------------------------------------------
# Fixed positions in the filename
#
#   0000021542_0000626880_ALOS41712610250622HBQ-RD0606_1.1__-.zip.0001
#   field[0]  field[1]   field[2] ...                  (split on '_')
#
#   field[2] = "ALOS41712610250622HBQ-RD0606"
#               0123456789012345678
#               ALOS4             -> [0:5]
#                       yymmdd    -> [12:18]  ("250622")
# ---------------------------------------------------------------------------
SCENE_FIELD_INDEX = 2
MISSION_PREFIX = "ALOS4"
DATE_START = 12
DATE_END = 18

TMP_SUFFIX = ".extracting"          # partially written file inside a dataset
CHUNK = 8 * 1024 * 1024             # read/write block size
REPORT_EVERY = 32 * 1024 * 1024     # worker sends progress after this many bytes


def parse_date_string(filename):
    """Return (date_string_yymmdd, date_object, error_message)."""
    fields = filename.split("_")
    if len(fields) <= SCENE_FIELD_INDEX:
        return None, None, "filename has too few '_' separated fields"
    scene = fields[SCENE_FIELD_INDEX]
    if not scene.startswith(MISSION_PREFIX):
        return None, None, "field %d (%s) does not start with %s" % (
            SCENE_FIELD_INDEX, scene, MISSION_PREFIX)
    if len(scene) < DATE_END:
        return None, None, "scene field '%s' too short to hold a date" % scene
    s = scene[DATE_START:DATE_END]
    if not s.isdigit():
        return s, None, "date string '%s' is not all digits" % s
    yy = int(s[0:2])
    mm = int(s[2:4])
    dd = int(s[4:6])
    try:
        d = datetime.date(2000 + yy, mm, dd)
    except ValueError as e:
        return s, None, "date string '%s' is not a valid yymmdd date (%s)" % (s, e)
    if d > datetime.date.today():
        return s, d, "date string '%s' (%s) is in the future" % (s, d.isoformat())
    if d < datetime.date(2024, 7, 1):  # ALOS-4 launched 1 July 2024
        return s, d, "date string '%s' (%s) is before ALOS-4 launch" % (s, d.isoformat())
    return s, d, None


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
def flat_name(name):
    """Last path component of a zip member: the folder levels inside the zip
    are dropped so every file goes directly into the date folder."""
    comps = []
    for c in name.replace("\\", "/").split("/"):
        if c == "" or c == "." or c == "..":
            continue
        comps.append(c)
    if not comps:
        raise ValueError("empty file name in zip: '%s'" % name)
    if comps[-1].endswith(TMP_SUFFIX):
        # would be mistaken for a half-written file and deleted on resume
        raise ValueError("file name in zip ends with reserved suffix '%s': %s" % (
            TMP_SUFFIX, name))
    return comps[-1]


def member_path(root, name):
    return os.path.join(root, flat_name(name))


def check_flat(members):
    """Error message if two files in the zip would get the same flat name."""
    seen = {}
    clashes = []
    for m in members:
        if m.is_dir():
            continue
        f = flat_name(m.filename)
        if f in seen:
            clashes.append("%s and %s" % (seen[f], m.filename))
        else:
            seen[f] = m.filename
    if clashes:
        return "same file name at different levels in zip: " + "; ".join(clashes)
    return None


def member_done(tmp_dir, m):
    """True if this member was already fully extracted in an earlier run."""
    if m.is_dir():
        return True
    p = member_path(tmp_dir, m.filename)
    return os.path.isfile(p) and os.path.getsize(p) == m.file_size


def fmt_hms(seconds):
    seconds = int(max(0, seconds))
    return "%02d:%02d:%02d" % (seconds // 3600, (seconds // 60) % 60, seconds % 60)


def fmt_gb(n):
    return "%.2f GB" % (n / 1e9)


# ---------------------------------------------------------------------------
# Find jobs
# ---------------------------------------------------------------------------
def find_parts(folder):
    """Return (sorted list of part paths, error or None) for one folder."""
    parts = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        dot = name.rfind(".")
        if dot < 0:
            continue
        base = name[:dot]
        suffix = name[dot + 1:]
        if not base.endswith(".zip") or not suffix.isdigit():
            continue
        parts.append((int(suffix), base, path))
    if not parts:
        return [], None
    parts.sort()
    bases = set(p[1] for p in parts)
    if len(bases) != 1:
        return [], "more than one dataset in folder: %s" % ", ".join(sorted(bases))
    numbers = [p[0] for p in parts]
    expected = list(range(1, len(parts) + 1))
    if numbers != expected:
        return [], "part numbers %s are not consecutive starting at 1" % numbers
    return [p[2] for p in parts], None


def build_jobs(top):
    jobs = []      # (out_name, src_name, [part paths])
    errors = []
    for name in sorted(os.listdir(top)):
        folder = os.path.join(top, name)
        if not os.path.isdir(folder):
            continue
        parts, err = find_parts(folder)
        if err:
            errors.append("%s: %s" % (name, err))
            continue
        if not parts:
            continue
        datestr, d, err = parse_date_string(os.path.basename(parts[0]))
        if err:
            errors.append("%s: %s" % (name, err))
            continue
        jobs.append(("20" + datestr, name, parts))

    # two different datasets must not land in the same date folder
    seen = {}
    unique = []
    for out_name, src_name, parts in jobs:
        if out_name in seen:
            errors.append("%s: same date folder %s as %s (not extracted)" % (
                src_name, out_name, seen[out_name]))
            continue
        seen[out_name] = src_name
        unique.append((out_name, src_name, parts))
    return unique, errors


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


def extract_job(top, out_name, src_name, parts):
    """Returns (src_name, out_name, status, message). status: ok/skip/error."""
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
        zf, buf = open_zip(parts)
        try:
            members = zf.infolist()
            err = check_flat(members)
            if err:
                raise ValueError(err)
            new_bytes = 0
            pending = 0
            for m in members:
                check_parent()
                if m.is_dir():
                    continue          # flat output: zip folders are not created
                dest = member_path(tmp_dir, m.filename)
                if member_done(tmp_dir, m):
                    continue
                tmp = dest + TMP_SUFFIX
                with zf.open(m) as src, open(tmp, "wb") as dst:
                    while True:
                        block = src.read(CHUNK)   # CRC is checked at end of member
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
            if pending:
                _progress_q.put(("bytes", out_name, pending))
                new_bytes += pending
        finally:
            zf.close()
            buf.close()

        os.rename(tmp_dir, final_dir)
        nfiles = sum(1 for m in members if not m.is_dir())
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
    ap = argparse.ArgumentParser(description="Extract ALOS-4 multi-part zips into 20yymmdd folders.")
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

    # --- pre-scan: sizes from each zip's table of contents, and what is
    #     already done from an earlier interrupted run
    total = {}       # out_name -> uncompressed bytes in dataset
    done = {}        # out_name -> bytes already extracted
    todo = []
    skipped = []
    for out_name, src_name, parts in jobs:
        if os.path.exists(os.path.join(top, out_name)):
            skipped.append((src_name, out_name))
            continue
        try:
            zf, buf = open_zip(parts)
            try:
                members = zf.infolist()
                err = check_flat(members)
                if err:
                    raise ValueError(err)
                tmp_dir = os.path.join(top, "." + out_name + ".partial")
                total[out_name] = sum(m.file_size for m in members)
                done[out_name] = 0
                if os.path.isdir(tmp_dir):
                    for m in members:
                        if not m.is_dir() and member_done(tmp_dir, m):
                            done[out_name] += m.file_size
            finally:
                zf.close()
                buf.close()
        except Exception as e:
            errors.append("%s -> %s: not extracted: %s: %s" % (
                src_name, out_name, type(e).__name__, e))
            continue
        todo.append((out_name, src_name, parts))

    grand_total = sum(total[j[0]] for j in todo)
    resumed = sum(done[j[0]] for j in todo)

    say("Directory: %s" % top)
    say("Workers: %d   Status every %g s" % (args.workers, args.interval))
    for src_name, out_name in skipped:
        say("  %s -> %s  already finished, skipped" % (src_name, out_name))
    for out_name, src_name, parts in todo:
        extra = ""
        if done[out_name]:
            extra = "  (resuming, %.1f%% already done)" % (
                100.0 * done[out_name] / max(1, total[out_name]))
        say("  %s -> %s  %d parts, %s%s" % (
            src_name, out_name, len(parts), fmt_gb(total[out_name]), extra))
    say("Datasets to process: %d   Total: %s%s" % (
        len(todo), fmt_gb(grand_total),
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
        for out_name, src_name, parts in todo:     # waiting, in queue order
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
            line = "[%s] %5.1f%%  %s / %s  |  %.1f MB/s  |  %s  |  datasets %d/%d done" % (
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
            for out_name, src_name, parts in todo:
                pool.apply_async(extract_job, (top, out_name, src_name, parts),
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
    if errors:
        say("ERRORS (%d):" % len(errors))
        for e in errors:
            say("  " + e)
        sys.exit(1)
    else:
        say("SUCCESS")


if __name__ == "__main__":
    main()
