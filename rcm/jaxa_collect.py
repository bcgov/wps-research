#!/usr/bin/env python3
"""
jaxa_collect.py -- gather one copy of every JAXA scene into one folder per area (moves by default)

USAGE
  cd /Volumes/d5t1/JAXA_COLLECTION && python3 jaxa_collect.py [--copy] [--dry-run] [--yes]

  python3 jaxa_collect.py             # plan, print every move command, ask, then MOVE the zips (instant, no new copies)
  python3 jaxa_collect.py --dry-run   # print the plan and the move commands only; change nothing
  python3 jaxa_collect.py --yes       # move without asking (the plan and commands are still printed first)
  python3 jaxa_collect.py --copy      # COPY with rsync instead (originals stay; needs the space; resumable, with ETA)
  python3 jaxa_collect.py --copy --dry-run   # copy plan and the space it needs; copy nothing
  sh jaxa_collect_undo.sh             # put every moved file back where it was (written by each move run)

WHAT IT NEEDS
  A folder called catalog/ in the present folder, holding the output of
  jaxa_catalog.py (JAXA_products_catalogue.csv and JAXA_scene_centres.dbf).
  Make it by running jaxa_catalog.py from inside catalog/ first, e.g.
    mkdir -p catalog && cd catalog && python3 jaxa_catalog.py --no-qgis /Volumes/d5t1 && cd ..
  Only JAXA product zips are considered: zips that merely have ALOS in their
  name (documents, photos, ...) are dropped first. Only JAXA copies on the
  same volume as the present folder are gathered; copies on other volumes
  or machines (e.g. in a --merge catalogue) are ignored, and scenes that
  exist only elsewhere are listed. On one volume a move only renames: no
  data is read or written, so even a terabyte takes seconds.

WHAT IT DOES
  1. Reads the catalogue, drops non-JAXA zips and copies on other volumes,
     and picks one undamaged copy per scene on this volume (one that passed
     --verify first, then the copy jaxa_catalog.py chose); takes the area
     name of each scene from NAME in JAXA_scene_centres.dbf (the top-level
     areas of JAXA_areas.shp).
  2. Named areas go to one folder each in the present folder (Elephant Hill
     -> Elephant_Hill/; areas sharing a name share a folder). JAXA scenes
     with no area name (an UNRESOLVED area, or no footprint) go to
     UNRESOLVED/. Scenes with no undamaged copy on this volume are listed
     and left where they are.
  3. Keeps one copy per image capture: copies with the same JAXA scene ID
     and the same size are the same product (e.g. the same scene in two
     orders), so only one is gathered. Products of one capture that differ
     in size (e.g. Level 1.1 and Level 1.5) are different products; both kept.
     Every part of a multipart zip goes along, flat, under its own file name.
  4. MOVE (default): prints the plan, every mkdir/mv command, and saves them
     as jaxa_collect_moves.sh; asks before moving (unless --yes); writes
     jaxa_collect_undo.sh (the reverse moves) before moving anything; then
     renames the files. Never overwrites: a file already at its destination
     with the right size is skipped, anything else there is reported.
     Duplicates, damaged and spare copies stay where they were; the run
     ends with a count of what was left behind (listed in the manifest).
  5. COPY (--copy): checks the free space first, then copies with rsync,
     one file at a time: a file already there at full size is skipped, a
     partial one is resumed (--append-verify where rsync has it). One status
     line for the whole collection: files, GB done / total, the current file,
     write speed, elapsed time, ETA and finish time.
  6. Writes jaxa_collect_manifest.csv (area, scene, source, destination,
     size, result, and every catalogued copy left in place, with the reason).

  Running it again is safe: files already gathered are recognised (also after
  a move, when the catalogue still lists the old paths) and skipped.
  Ctrl-C stops cleanly; run again to carry on where it stopped.
  Files already in a folder of an earlier collection here are never moved
  again. Nothing is ever deleted.
  After moving, keep catalog/ as it is: it lists the old locations and this
  script recognises the moved files with it. A fresh jaxa_catalog.py scan of
  the collection would lose the area names that came from the original
  folder names (e.g. 2021_Elephant_Hill, EODMS_ALOS2_FP6_DSC).

CHANGES
  2026-11-14  first version
  2026-11-16  moves by default (--copy for rsync); plan of move commands shown and
              confirmed first; undo script; unresolved JAXA scenes -> UNRESOLVED/;
              reruns recognise files already moved; leftovers reported;
              non-JAXA zips dropped first; copies on other volumes ignored (merged
              catalogues work), best undamaged copy on this volume chosen per scene;
              every list printed in full (with sizes, parts and reasons)
"""
import csv
import os
import shlex
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
from collections import OrderedDict, deque

CATALOG_DIR = 'catalog'
CSV_FILE = 'JAXA_products_catalogue.csv'
CENTRES_DBF = 'JAXA_scene_centres.dbf'
MANIFEST = 'jaxa_collect_manifest.csv'
MOVES_SH = 'jaxa_collect_moves.sh'
UNDO_SH = 'jaxa_collect_undo.sh'
UNRESOLVED_DIR = 'UNRESOLVED'         # JAXA scenes with no area name
UNDAMAGED = ('ok', 'no_footprint')
SPACE_MARGIN = 2_000_000_000         # keep this much free after copying
RATE_WINDOW_S = 20                    # write speed = bytes written over the last 20 s
STATUS_EVERY_S = 1.0                  # status line refresh (terminal)
LOG_EVERY_S = 60.0                    # status line period when output is not a terminal
RSYNC_CANDIDATES = ['/opt/local/bin/rsync', '/opt/homebrew/bin/rsync', '/usr/local/bin/rsync']

csv.field_size_limit(sys.maxsize)
TTY = sys.stdout.isatty()


def die(msg):
    sys.stdout.write('\n' if TTY else '')
    sys.exit(f"jaxa_collect.py: {msg}")


def gb(n):
    if n >= 1e9 or n == 0:
        return f"{n / 1e9:,.1f} GB"
    return f"{n / 1e6:,.1f} MB" if n >= 1e6 else f"{n / 1e3:,.1f} kB"


def hms(s):
    s = int(max(0, s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


# ----------------------------------------------------------------------------- catalogue

def read_dbf(path):
    """Minimal dBase III reader (all fields as stripped text, UTF-8)."""
    b = open(path, 'rb').read()
    n, hlen, rlen = struct.unpack('<IHH', b[4:12])
    fields, o = [], 32
    while o < hlen and b[o] != 0x0D:
        fields.append((b[o:o + 11].split(b'\0')[0].decode('ascii', 'replace'), b[o + 16]))
        o += 32
    out = []
    for i in range(n):
        r = b[hlen + i * rlen: hlen + (i + 1) * rlen]
        if r[:1] == b'*':                      # deleted record
            continue
        d, p = {}, 1
        for name, ln in fields:
            d[name] = r[p:p + ln].decode('utf-8', 'replace').strip()
            p += ln
        out.append(d)
    return out


def load_catalogue():
    csv_path = os.path.join(CATALOG_DIR, CSV_FILE)
    dbf_path = os.path.join(CATALOG_DIR, CENTRES_DBF)
    missing = [p for p in (csv_path, dbf_path) if not os.path.isfile(p)]
    if missing:
        die("no jaxa_catalog.py output found here:\n  missing " + "\n  missing ".join(missing) + "\n\n"
            "jaxa_catalog.py has to be run first, with its output in a folder called\n"
            f"'{CATALOG_DIR}' inside the present folder ({os.getcwd()}), e.g.\n"
            f"  mkdir -p {CATALOG_DIR} && cd {CATALOG_DIR} && python3 jaxa_catalog.py --no-qgis {os.getcwd()} && cd ..\n"
            "(or scan the whole volume this folder is on), then run jaxa_collect.py again from here.")
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    need = {'is_representative', 'status', 'scene_id', 'zip_path', 'all_parts', 'part_sizes', 'zip_size'}
    if not rows or need - set(rows[0]):
        die(f"{csv_path} has no rows or is missing columns {sorted(need - set(rows[0] if rows else {}))};\n"
            "it was made by an older jaxa_catalog.py. Run the current jaxa_catalog.py again into catalog/.")
    centres = read_dbf(dbf_path)
    if centres and not {'NAME', 'CSV_ROW', 'ZIP'} <= set(centres[0]):
        die(f"{dbf_path} has no NAME / CSV_ROW / ZIP fields; it was made by an older jaxa_catalog.py.\n"
            "Run the current jaxa_catalog.py again into catalog/.")
    names, out_of_sync = {}, []
    for c in centres:
        try:
            i = int(c['CSV_ROW']) - 1
        except ValueError:
            continue
        zp = rows[i]['zip_path'] if 0 <= i < len(rows) else None
        # the DBF keeps at most 254 characters of ZIP
        if zp is None or not (zp == c['ZIP'] or (len(c['ZIP']) >= 250 and zp.startswith(c['ZIP']))):
            out_of_sync.append(c['ZIP'])
            continue
        names[i] = c['NAME']
    if out_of_sync:
        die(f"{CENTRES_DBF} and {CSV_FILE} in catalog/ do not belong together "
            f"({len(out_of_sync)} scene(s) point at different rows, e.g. {out_of_sync[0]}).\n"
            "Copy both from the same jaxa_catalog.py run (or run it again into catalog/).")
    return rows, names


def row_parts(r):
    parts = [p for p in r.get('all_parts', '').split(';') if p] or [r['zip_path']]
    sizes = [s for s in r.get('part_sizes', '').split(';') if s]
    sizes = [int(s) for s in sizes] if len(sizes) == len(parts) else [None] * len(parts)
    return list(zip(parts, sizes))


def device_of(path):
    """st_dev of the path, or of its nearest existing parent (for deleted files)."""
    p = os.path.abspath(path)
    while True:
        try:
            return os.stat(p).st_dev
        except OSError:
            q = os.path.dirname(p)
            if q == p:
                return None
            p = q


def is_jaxa(r):
    """A JAXA product zip: its metadata was read (ok / no_footprint / damaged but readable),
    or a JAXA scene ID was found. Other zips with ALOS in the name are not."""
    st = r.get('status', '')
    if st == 'not_jaxa_product':
        return False
    return st in UNDAMAGED + ('damaged_recovered',) or bool(r.get('scene_id'))


def on_this_volume(r, here_dev):
    return all(device_of(p) == here_dev for p, _ in row_parts(r))


def screen(rows):
    """Drops non-JAXA zips first, then sorts the JAXA copies into this volume / elsewhere.
    Returns (indices of JAXA rows on this volume, n non-JAXA, n JAXA copies elsewhere)."""
    here_dev = os.stat('.').st_dev
    local, n_non, n_else = [], 0, 0
    for i, r in enumerate(rows):
        if not is_jaxa(r):
            n_non += 1
        elif on_this_volume(r, here_dev):
            local.append(i)
        else:
            n_else += 1
    if not local:
        die(f"none of the JAXA zips in catalog/ are on the volume of the present folder ({os.getcwd()}).\n"
            "Run jaxa_collect.py from a folder on the volume that holds the JAXA zips, with a catalogue\n"
            "that covers that volume (jaxa_catalog.py run into catalog/).")
    return local, n_non, n_else


def describe(r):
    """'path (size, N parts)' for listings."""
    parts = row_parts(r)
    size = sum(s or 0 for _, s in parts)
    extra = f", {len(parts)} parts" if len(parts) > 1 else ''
    return f"{r['zip_path']}  ({gb(size) if size else 'size unknown'}{extra})"


def safe_folder(name):
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')[:60] or 'area'


# ----------------------------------------------------------------------------- plan

def make_plan(rows, names, local):
    """Returns the files to gather and everything left out, with reasons.
    Each candidate is (row index, row, area name, folder name)."""
    here = os.path.realpath('.')
    skipped = OrderedDict((k, []) for k in ('no undamaged copy on this volume (re-download)',
                                            'scene only on other volumes / machines'))
    cands, unresolved, spares = [], [], []
    local = set(local)
    scene_rows = OrderedDict()                   # scene_key -> indices of its JAXA copies (all volumes)
    for i, r in enumerate(rows):
        if is_jaxa(r):
            scene_rows.setdefault(r.get('scene_key') or r.get('product') or os.path.basename(r['zip_path']),
                                  []).append(i)

    def usable(r):
        return r.get('status', '') in UNDAMAGED and not r.get('verify_status', '').startswith('FAILED')

    def in_folder(r, folder):
        d = os.path.join(here, folder)
        return all(os.path.isfile(os.path.join(d, os.path.basename(p))) for p, _ in row_parts(r))

    for key, idx in scene_rows.items():
        name = next((names[i] for i in idx if names.get(i)), '')
        if not name:
            name, folder = 'UNRESOLVED (no footprint)', UNRESOLVED_DIR
        elif name.upper().startswith('UNRESOLVED'):
            folder = UNRESOLVED_DIR
        else:
            folder = safe_folder(name)
        mine = [i for i in idx if i in local]
        ok = [i for i in mine if usable(rows[i])]
        if not ok:
            if mine:
                skipped['no undamaged copy on this volume (re-download)'].append(rows[mine[0]])
            else:
                skipped['scene only on other volumes / machines'].append(rows[idx[0]])
            continue
        ok.sort(key=lambda i: (not rows[i].get('verify_status', '').startswith('passed'),
                               not in_folder(rows[i], folder), rows[i].get('is_representative') != 'yes',
                               rows[i]['zip_path']))
        cands.append((ok[0], rows[ok[0]], name, folder))
        spares += [rows[i] for i in ok[1:]]
        if folder == UNRESOLVED_DIR:
            unresolved.append(rows[ok[0]])

    # one copy per capture: same scene ID and same size = same product
    groups = OrderedDict()
    for c in cands:
        r = c[1]
        key = (r.get('scene_id') or r.get('scene_key') or r.get('product') or r['zip_path'], r.get('zip_size', ''))
        groups.setdefault(key, []).append(c)
    scenes, duplicates, size_variants = [], [], {}

    def already_here(item):
        i, r, name, folder = item
        d = os.path.join(here, folder)
        return all(os.path.isfile(os.path.join(d, os.path.basename(p))) and
                   (s is None or os.path.getsize(os.path.join(d, os.path.basename(p))) == s)
                   for p, s in row_parts(r))

    for key, g in groups.items():
        g.sort(key=lambda it: (not it[1].get('verify_status', '').startswith('passed'),
                               not already_here(it), it[1]['zip_path']))
        keep = g[0]
        for other in g[1:]:
            duplicates.append((other[1], keep[1]))
        scenes.append(keep)
        if key[0]:
            size_variants.setdefault(key[0], []).append(keep[1])
    size_variants = {k: v for k, v in size_variants.items() if len(v) > 1 and k.startswith('ALOS')}

    # files to copy
    files, taken = [], {}
    problems = []
    for i, r, name, fname in scenes:
        folder = os.path.join(here, fname)
        for src, size in row_parts(r):
            dst = os.path.join(folder, os.path.basename(src))
            if dst in taken and taken[dst] != src:
                problems.append(f"two different sources for {dst}: {taken[dst]} and {src} (second skipped)")
                continue
            taken[dst] = src
            have = os.path.getsize(dst) if os.path.isfile(dst) else 0
            entry = {'area': name, 'fname': fname, 'folder': folder, 'src': src, 'dst': dst, 'size': size,
                     'have': have, 'same': False, 'gone': False, 'row': r, 'result': ''}
            if not os.path.exists(src):
                if os.path.isfile(dst) and (size is None or have == size):
                    entry.update(size=have, gone=True)      # moved here by an earlier run
                    files.append(entry)
                else:
                    problems.append(f"missing source (catalogue out of date? run jaxa_catalog.py again): {src}")
                continue
            real = os.path.getsize(src)
            if size is not None and real != size:
                problems.append(f"size changed since cataloguing ({size:,} -> {real:,} B): {src}")
                continue
            entry['size'] = real
            if os.path.exists(dst) and os.path.samefile(src, dst):
                entry.update(same=True, have=real)
            elif have > real:
                problems.append(f"destination larger than its source ({have:,} > {real:,} B), not touched; "
                                f"check it and delete it, then run again: {dst}")
                continue
            files.append(entry)
    return files, scenes, skipped, unresolved, duplicates, size_variants, problems, spares


def left_behind(rows, local, files, skipped, duplicates, spares):
    """JAXA copies on this volume that are not gathered, with the reason: (row, reason)."""
    used = {id(f['row']) for f in files}
    why = {id(r): 'spare copy of a gathered scene' for r in spares}
    for k, rs in skipped.items():
        for r in rs:
            why[id(r)] = k
    for r, keep in duplicates:
        why[id(r)] = f"duplicate of {os.path.basename(keep['zip_path'])}"
    out = []
    for r in (rows[i] for i in local):
        if id(r) in used:
            continue
        if id(r) in why:
            reason = why[id(r)]
        elif r.get('status', '') in UNDAMAGED and not r.get('verify_status', '').startswith('FAILED'):
            reason = 'not gathered (see PROBLEMS)'
        else:
            reason = f"damaged copy ({r.get('status', '')}{', verify FAILED' if r.get('verify_status', '').startswith('FAILED') else ''})"
        out.append((r, reason))
    return out


# ----------------------------------------------------------------------------- rsync

def pick_rsync():
    """The rsync with the best resume option: --append-verify > --append > --inplace."""
    cands = []
    w = shutil.which('rsync')
    for p in ([w] if w else []) + RSYNC_CANDIDATES:
        if p and os.access(p, os.X_OK) and os.path.realpath(p) not in [os.path.realpath(c) for c in cands]:
            cands.append(p)
    best = None
    for p in cands:
        try:
            out = subprocess.run([p, '--help'], capture_output=True, text=True, timeout=20)
            txt = out.stdout + out.stderr
            ver = subprocess.run([p, '--version'], capture_output=True, text=True, timeout=20)
            vtxt = (ver.stdout + ver.stderr).splitlines()[0] if (ver.stdout + ver.stderr) else p
        except (OSError, subprocess.SubprocessError):
            continue
        if '--append-verify' in txt:
            mode, flags = 3, ['--append-verify']
        elif '--append' in txt:
            mode, flags = 2, ['--append']
        elif '--inplace' in txt:
            mode, flags = 1, ['--inplace', '--partial']
        else:
            mode, flags = 0, ['--partial']
        if best is None or mode > best[0]:
            best = (mode, p, flags, vtxt.strip())
    if best is None:
        die("rsync not found. Install it (macOS: 'sudo port install rsync' or 'brew install rsync';\n"
            "Ubuntu: 'sudo apt install rsync') and run again.")
    return best[1], ['-t'] + best[2], best[3], best[0]


def written_size(dst):
    """Bytes of dst written so far, including rsync's temporary .name.XXXXXX file."""
    n = 0
    try:
        n = os.path.getsize(dst)
    except OSError:
        pass
    d, base = os.path.split(dst)
    try:
        for f in os.listdir(d):
            if f.startswith('.' + base + '.'):
                try:
                    n = max(n, os.path.getsize(os.path.join(d, f)))
                except OSError:
                    pass
    except OSError:
        pass
    return n


class Status:
    def __init__(self, files):
        self.files = files
        self.total = sum(f['size'] for f in files)
        self.n = len(files)
        self.start = time.time()
        self.samples = deque()                   # (time, bytes written this session)
        self.session = 0                         # bytes written this session, finished files
        self.last_draw = 0.0
        self.width = 0

    def done_bytes(self, cur=None, cur_written=0):
        b = sum(f['size'] for f in self.files if f['result'] in ('copied', 'already there', 'in place'))
        if cur is not None:
            b += cur_written
        return b

    def line(self, idx, cur=None, cur_written=0, phase=''):
        now = time.time()
        written = self.session + (max(0, cur_written - cur['have']) if cur else 0)
        self.samples.append((now, written))
        while len(self.samples) > 2 and now - self.samples[0][0] > RATE_WINDOW_S:
            self.samples.popleft()
        t0, b0 = self.samples[0]
        rate = (written - b0) / (now - t0) if now - t0 >= 2 else 0.0
        elapsed = now - self.start
        avg = written / elapsed if elapsed > 0 else 0.0
        done = self.done_bytes(cur, cur_written)
        left = max(0, self.total - done)
        r = rate if rate > 0 else avg
        eta = left / r if r > 0 else None
        files_done = sum(1 for f in self.files if f['result'] in ('copied', 'already there', 'in place'))
        s = (f"[{files_done}/{self.n} files] {done / 1e9:,.1f}/{self.total / 1e9:,.1f} GB "
             f"({100 * done / self.total if self.total else 100:.1f}%), {left / 1e9:,.1f} GB left")
        if cur is not None:
            pct = 100 * cur_written / cur['size'] if cur['size'] else 100
            s += f" | file {idx + 1}: {pct:3.0f}%{(' ' + phase) if phase else ''} @ {rate / 1e6:,.1f} MB/s"
        s += f" | elapsed {hms(elapsed)}"
        if eta is not None and left:
            fin = time.localtime(now + eta)
            fmt = '%H:%M' if eta < 20 * 3600 else '%a %H:%M'
            s += f" | ETA {hms(eta)} (done ~{time.strftime(fmt, fin)})"
        elif left:
            s += " | ETA --"
        return s

    def draw(self, text, force=False):
        now = time.time()
        if TTY:
            if not force and now - self.last_draw < STATUS_EVERY_S:
                return
            cols = shutil.get_terminal_size((160, 20)).columns - 1
            text = text[:cols]
            sys.stdout.write('\r' + text + ' ' * max(0, self.width - len(text)))
            sys.stdout.flush()
            self.width = len(text)
        else:
            if not force and now - self.last_draw < LOG_EVERY_S:
                return
            print(text, flush=True)
        self.last_draw = now

    def say(self, text):
        """Print a message line above the status line."""
        if TTY:
            sys.stdout.write('\r' + ' ' * self.width + '\r')
            self.width = 0
        print(text, flush=True)


def copy_all(files, rsync, flags, mode):
    st = Status(files)
    stopping = {'flag': False}
    proc = {'p': None}

    def on_int(sig, frm):
        stopping['flag'] = True
        if proc['p'] and proc['p'].poll() is None:
            proc['p'].terminate()
    old = signal.signal(signal.SIGINT, on_int)
    errlog = os.path.join(os.getcwd(), '.jaxa_collect_rsync.err')
    try:
        n_pre = sum(1 for f in files if f['result'])
        if n_pre:
            st.say(f"{n_pre} file(s) already complete in the area folders ({gb(st.done_bytes())}), skipped.")
        for idx, f in enumerate(files):
            if f['result']:
                continue
            if stopping['flag']:
                break
            os.makedirs(f['folder'], exist_ok=True)
            have = written_size(f['dst'])
            if have > f['size']:
                f['result'] = 'FAILED: destination is larger than the source; check it and delete it'
                st.say(f"  !! {f['dst']} is larger than its source ({have:,} > {f['size']:,} B); left alone")
                continue
            f['have'] = have
            label = os.path.relpath(f['dst'])
            st.say(f"  {'resume' if have else 'copy  '} {label}  ({gb(f['size'])}"
                   + (f", {gb(have)} already there" if have else '') + ")")
            with open(errlog, 'w') as err:
                p = subprocess.Popen([rsync] + flags + [f['src'], f['dst']],
                                     stdout=subprocess.DEVNULL, stderr=err, stdin=subprocess.DEVNULL,
                                     start_new_session=True)
                proc['p'] = p
                while p.poll() is None:
                    w = written_size(f['dst'])
                    phase = 'checking' if w >= f['size'] else ''
                    st.draw(st.line(idx, f, w, phase))
                    try:
                        p.wait(timeout=0.25)
                    except subprocess.TimeoutExpired:
                        pass
            proc['p'] = None
            w = written_size(f['dst'])
            if p.returncode == 0 and os.path.isfile(f['dst']) and os.path.getsize(f['dst']) == f['size']:
                f['result'] = 'copied'
                st.session += f['size'] - f['have']
                try:                                   # a resumed file can keep rsync's 0600 temp mode
                    shutil.copymode(f['src'], f['dst'])
                except OSError:
                    pass
            else:
                msg = open(errlog, errors='replace').read().strip().splitlines()
                if stopping['flag']:
                    f['result'] = f"stopped at {gb(w)} of {gb(f['size'])} (resumes next run)"
                    st.session += max(0, w - f['have'])
                    break
                f['result'] = f"FAILED: rsync exit {p.returncode}" + (f": {msg[-1]}" if msg else '')
                st.session += max(0, w - f['have'])
                st.say(f"  !! {label}: {f['result']}")
            st.draw(st.line(idx), force=True)
    finally:
        signal.signal(signal.SIGINT, old)
        try:
            os.remove(errlog)
        except OSError:
            pass
    if TTY:
        sys.stdout.write('\n')
    return st, stopping['flag']


# ----------------------------------------------------------------------------- move

def q(p):
    return shlex.quote(p)


def move_commands(todo, here):
    """Shell commands for the plan: mkdir -p per new folder, then mv -n per file."""
    cmds, made = [], set()
    for f in todo:
        if f['folder'] not in made and not os.path.isdir(f['folder']):
            cmds.append(f"mkdir -p {q(os.path.relpath(f['folder'], here))}")
        made.add(f['folder'])
        cmds.append(f"mv -n {q(f['src'])} {q(os.path.relpath(f['dst'], here))}")
    return cmds


def write_script(path, lines, comment):
    with open(path, 'w', encoding='utf-8') as fo:
        fo.write('#!/bin/sh\n# ' + comment + '\n# written by jaxa_collect.py ' + time.strftime('%Y-%m-%d %H:%M')
                 + ' in ' + os.getcwd() + '\n' + f"cd {q(os.getcwd())} || exit 1\n" + '\n'.join(lines) + '\n')
    os.chmod(path, 0o755)


def move_all(files):
    """Rename every file (same volume: no data copied). Never overwrites."""
    stopping = {'flag': False}

    def on_int(sig, frm):
        stopping['flag'] = True
    old = signal.signal(signal.SIGINT, on_int)
    todo = [f for f in files if not f['result']]
    t0 = time.time()
    try:
        for k, f in enumerate(todo, 1):
            if stopping['flag']:
                break
            try:
                os.makedirs(f['folder'], exist_ok=True)
                if os.path.lexists(f['dst']):
                    raise OSError(f"{f['dst']} appeared meanwhile; not overwritten")
                os.rename(f['src'], f['dst'])
                f['result'] = 'moved'
            except OSError as e:
                f['result'] = f"FAILED: {e}"
                print(f"\n  !! {f['src']}: {f['result']}")
            if TTY:
                sys.stdout.write(f"\r  moved {sum(1 for x in todo if x['result'] == 'moved')}/{len(todo)} file(s)")
                sys.stdout.flush()
    finally:
        signal.signal(signal.SIGINT, old)
    if TTY:
        sys.stdout.write('\n')
    return time.time() - t0, stopping['flag']


def write_manifest(files, leftovers):
    try:
        with open(MANIFEST, 'w', newline='', encoding='utf-8') as fo:
            wr = csv.writer(fo)
            wr.writerow(['area', 'folder', 'scene_id', 'product_id', 'acq_center', 'source', 'destination',
                         'size', 'result'])
            for f in files:
                r = f['row']
                wr.writerow([f['area'], f['fname'], r.get('scene_id', ''), r.get('product_id', ''),
                             r.get('acq_center', ''), f['src'], os.path.relpath(f['dst']), f['size'],
                             f['result'] or 'not started'])
            for r, reason in leftovers:
                for p, s in row_parts(r):
                    wr.writerow(['', '', r.get('scene_id', ''), r.get('product_id', ''), r.get('acq_center', ''),
                                 p, '', s if s is not None else '', f"left in place: {reason}"])
    except OSError as e:
        print(f"could not write {MANIFEST}: {e}")


def ask(question):
    if not sys.stdin.isatty():
        die("no terminal to ask for confirmation: run with --yes to go ahead, or --dry-run to look only.")
    try:
        return input(question).strip().lower() in ('y', 'yes')
    except EOFError:
        return False


# ----------------------------------------------------------------------------- main

def main():
    args = sys.argv[1:]
    if any(a in ('-h', '--help') for a in args):
        print(__doc__.strip())
        return
    bad = [a for a in args if a not in ('--dry-run', '--copy', '--yes', '-y')]
    if bad:
        die(f"unknown option(s): {' '.join(bad)}  (see -h)")
    dry, copy, yes = '--dry-run' in args, '--copy' in args, ('--yes' in args or '-y' in args)
    verb = 'copy' if copy else 'move'

    print(f"jaxa_collect.py in {os.getcwd()}  [{'COPY with rsync' if copy else 'MOVE'} mode"
          f"{', dry run' if dry else ''}]")
    rows, names = load_catalogue()
    local, n_non, n_else = screen(rows)
    print(f"catalog/: {len(rows)} catalogued zip(s): {n_non} not JAXA products (ignored), "
          f"{len(local) + n_else} JAXA copies, {len(local)} of them on this volume"
          + (f" ({n_else} on other volumes / machines, ignored)" if n_else else ''))

    non_jaxa = [r for r in rows if not is_jaxa(r)]
    if non_jaxa:
        print(f"ignored, not JAXA products: {len(non_jaxa)}")
        for r in sorted(non_jaxa, key=lambda r: r['zip_path']):
            print(f"    {describe(r)}  [{r.get('status', '') or 'no status'}]")

    files, scenes, skipped, unresolved, dups, variants, problems, spares = make_plan(rows, names, local)
    leftovers = left_behind(rows, local, files, skipped, dups, spares)

    for why, items in skipped.items():
        if items:
            print(f"not gathered, {why}: {len(items)} scene(s)")
            for r in items:
                print(f"    {describe(r)}")
    if unresolved:
        print(f"JAXA scenes with no area name -> {UNRESOLVED_DIR}/: {len(unresolved)}")
        for r in unresolved:
            print(f"    {describe(r)}")
    if dups:
        print(f"left where they are, same scene ID and size as the copy gathered (duplicate order/file): {len(dups)}")
        for a, b in dups:
            print(f"    {a['zip_path']}\n      = {b['zip_path']}")
    if variants:
        print(f"kept, one capture with products of different sizes (e.g. other processing level): {len(variants)}")
        for k, v in variants.items():
            print(f"    {k}: " + ', '.join(f"{r.get('product_id') or '?'} ({gb(int(r['zip_size'] or 0))})" for r in v))
    if problems:
        print(f"PROBLEMS ({len(problems)}), these files are left out:")
        for x in problems:
            print(f"    {x}")

    if not files:
        print("Nothing to collect.")
        return

    # folders of an earlier collection here: files in them are never moved/copied again, even if
    # a newer catalogue puts them in another area (a fresh jaxa_catalog.py scan of the collection
    # cannot see names that came from folder names such as 2021_Elephant_Hill)
    here = os.path.realpath('.')
    coll = {UNRESOLVED_DIR} | {f['fname'] for f in files}
    try:
        with open(MANIFEST, newline='', encoding='utf-8') as fi:
            coll |= {r['folder'] for r in csv.DictReader(fi) if r.get('folder')}
    except (OSError, KeyError, csv.Error):
        pass
    coll = {os.path.join(here, c) for c in coll}
    kept = []

    # files already complete at their destination count as done
    for f in files:
        sdir = os.path.dirname(os.path.realpath(f['src'])) if not f['gone'] else ''
        if f['same']:
            f['result'] = 'in place'
        elif sdir in coll and sdir != os.path.realpath(f['folder']):
            f['result'] = 'in place'
            kept.append((f, os.path.basename(sdir)))
            catalogue_folder = f['fname']
            f.update(fname=os.path.basename(sdir), folder=sdir, dst=f['src'], was=catalogue_folder)
        elif f['gone'] or f['have'] == f['size']:
            f['result'] = 'already there'
        elif not copy and f['have']:
            f['result'] = (f"FAILED: a different, smaller file is already at the destination "
                           f"({f['have']:,} of {f['size']:,} B, e.g. an unfinished --copy); finish it with "
                           f"--copy, or delete it and run again")
    if kept:
        print(f"already collected in another folder here, left there: {len(kept)}")
        for f, d in kept:
            print(f"    {d}/{os.path.basename(f['src'])}  (catalogue now says {f['was']}/)")
    blocked = [f for f in files if f['result'].startswith('FAILED')]
    for f in blocked:
        problems.append(f"{f['dst']}: {f['result'][8:]}")
        print(f"PROBLEM: {f['dst']}: {f['result'][8:]}")

    # plan per folder
    per = OrderedDict()
    for f in sorted(files, key=lambda f: (f['fname'] == UNRESOLVED_DIR, f['fname'].lower())):
        a = per.setdefault(f['fname'], {'scenes': set(), 'files': 0, 'size': 0, 'todo': 0, 'nfiles': 0})
        a['scenes'].add(id(f['row']))
        a['files'] += 1
        a['size'] += f['size']
        if not f['result']:
            a['nfiles'] += 1
            a['todo'] += f['size'] - (f['have'] if copy else 0)
    total = sum(a['size'] for a in per.values())
    todo_bytes = sum(a['todo'] for a in per.values())
    todo = [f for f in files if not f['result']]
    files.sort(key=lambda f: (f['fname'] == UNRESOLVED_DIR, f['fname'].lower(), f['dst']))
    todo.sort(key=lambda f: (f['fname'] == UNRESOLVED_DIR, f['fname'].lower(), f['dst']))
    print(f"\nPLAN: {len(scenes)} scene(s), {len(files)} file(s), {gb(total)} into {len(per)} folder(s) here:")
    w = max(len(k) for k in per) + 1
    print(f"    {'folder':<{w}} scenes  files        size   to {verb}")
    for k, a in per.items():
        print(f"    {k + '/':<{w}} {len(a['scenes']):>6} {a['files']:>6} {gb(a['size']):>11} "
              f"{a['nfiles']:>5} file(s), {gb(a['todo'])}")
    n_pre = sum(1 for f in files if f['result'] in ('in place', 'already there'))
    if n_pre:
        print(f"    ({n_pre} file(s) already in their folder, skipped)")
    left_size = sum(sum(s or 0 for _, s in row_parts(r)) for r, _ in leftovers)
    print(f"Left where they are on this volume: {len(leftovers)} JAXA copies ({gb(left_size)}): duplicates, "
          f"damaged and spare copies (also listed in {MANIFEST} after a run)")
    for r, reason in sorted(leftovers, key=lambda x: (x[1], x[0]['zip_path'])):
        print(f"    {describe(r)}  [{reason}]")

    if not todo:
        if blocked:
            print(f"\nNothing to {verb} apart from the {len(blocked)} problem file(s) above; "
                  f"{len(files) - len(blocked)} file(s) already collected.")
        else:
            print(f"\nEverything is already collected ({len(files)} file(s), {gb(total)}).")
        if not dry:
            write_manifest(files, leftovers)
        return

    # ------------------------------------------------------------ move mode
    if not copy:
        here = os.getcwd()
        cmds = move_commands(todo, here)
        print(f"\nMOVE COMMANDS ({len(todo)} file(s), {gb(todo_bytes)}; same volume, so these only rename,"
              f" no data is copied):")
        for c in cmds:
            print('  ' + c)
        try:
            write_script(MOVES_SH, cmds, 'the moves planned by jaxa_collect.py (for reference; the script does them itself)')
            print(f"(saved as {MOVES_SH})")
        except OSError as e:
            print(f"could not write {MOVES_SH}: {e}")
        if dry:
            print("\n--dry-run: nothing moved.")
            return
        if not yes and not ask(f"\nMove these {len(todo)} file(s) now? [y/N] "):
            print("Nothing moved.")
            return
        undo = []
        for f in reversed(todo):
            undo.append(f"mkdir -p {q(os.path.dirname(f['src']))} && mv -n {q(os.path.relpath(f['dst'], here))} {q(f['src'])}")
        try:
            write_script(UNDO_SH, undo, 'puts the files moved by jaxa_collect.py back where they were')
        except OSError as e:
            die(f"could not write {UNDO_SH} ({e}); nothing moved.")
        el, stopped = move_all(files)
        write_manifest(files, leftovers)
        moved = [f for f in files if f['result'] == 'moved']
        failed = [f for f in files if f['result'].startswith('FAILED')]
        notdone = [f for f in files if not f['result']]
        print(f"\n{'STOPPED' if stopped else 'DONE'} in {el:.1f} s: {len(moved)} file(s) moved "
              f"({gb(sum(f['size'] for f in moved))}), {n_pre} already there, {len(failed)} failed, "
              f"{len(notdone)} not done.")
        for f in failed:
            print(f"    {f['src']}: {f['result']}")
        print(f"Manifest: {MANIFEST}   Undo: sh {UNDO_SH}")
        if moved:
            print("Keep catalog/ as it is: it lists the old locations, and jaxa_collect.py recognises the moved\n"
                  "files with it. (A fresh jaxa_catalog.py scan of the new layout would lose area names that\n"
                  "came from folder names such as 2021_Elephant_Hill or EODMS_ALOS2_FP6_DSC.)")
        if failed:
            sys.exit(1)
        if stopped or notdone:
            print("Run jaxa_collect.py again to carry on.")
            sys.exit(130)
        return

    # ------------------------------------------------------------ copy mode
    free = shutil.disk_usage('.').free
    print(f"\nSPACE: {gb(todo_bytes)} still to copy, {gb(free)} free on this volume "
          f"(keeping {gb(SPACE_MARGIN)} spare)")
    if todo_bytes + SPACE_MARGIN > free:
        die(f"not enough space: {gb(todo_bytes + SPACE_MARGIN - free)} short. Nothing was copied.")

    rsync, flags, version, mode = pick_rsync()
    resume = {3: 'partial files resumed and checksum-verified (--append-verify)',
              2: 'partial files resumed (--append)', 1: 'partial files rewritten in place (--inplace)',
              0: 'partial files kept (--partial); this rsync cannot resume, consider a newer rsync'}[mode]
    print(f"RSYNC: {rsync} ({version}); {resume}")

    if dry:
        print("\n--dry-run: nothing copied.")
        return
    print()
    t0 = time.time()
    st, stopped = copy_all(files, rsync, flags, mode)
    write_manifest(files, leftovers)

    el = time.time() - t0
    copied = [f for f in files if f['result'] == 'copied']
    failed = [f for f in files if f['result'].startswith('FAILED')]
    pre = [f for f in files if f['result'] in ('already there', 'in place')]
    left = [f for f in files if not f['result'] or f['result'].startswith('stopped')]
    print(f"\n{'STOPPED' if stopped else 'DONE'} after {hms(el)}: {len(copied)} file(s) copied "
          f"({gb(st.session)} written, {st.session / el / 1e6 if el else 0:,.1f} MB/s average), "
          f"{len(pre)} already there, {len(failed)} failed, {len(left)} not done.")
    for f in failed:
        print(f"    {os.path.relpath(f['dst'])}: {f['result']}")
    if stopped or left:
        print("Run jaxa_collect.py --copy again to carry on; complete files are skipped, the partial one is resumed.")
    print(f"Manifest: {MANIFEST}")
    if failed:
        sys.exit(1)
    if stopped:
        sys.exit(130)


if __name__ == '__main__':
    main()
