#!/usr/bin/env python3
"""
20261101 snap_install.py  --  find, verify, install and update ESA SNAP

Makes sure two ESA SNAP executables are available:

    gpt        SNAP's command-line Graph Processing Tool
    esa-snap   SNAP's GUI. ESA ships it as "bin/snap"; it is renamed here so it
               can never be confused with Ubuntu's "snap" package manager.

What it does, in order:

 1. Looks for existing SNAP installs in the locations used by ESA's installer
    (old and new defaults), common manual locations, the user's PATH and the
    usual symlink folders. It does not scan the whole file system.
 2. Checks every candidate by running it with its help option and inspecting
    the output, so that e.g. Ubuntu's "snap" or a partition tool called "gpt"
    is never mistaken for SNAP.
 3. If no working install is found: downloads the latest "Linux 64-bit, All
    Toolboxes" installer from step.esa.int into the current directory,
    verifies its SHA-256 checksum, and installs it unattended into
    ~/esa-snap. Then renames bin/snap to bin/esa-snap.
 4. If a newer SNAP release, or module updates, are available, asks y/N
    before installing them.
 5. Adds the SNAP bin folder to the end of PATH in ~/.bashrc, so that
    system tools with the same names (e.g. another "gpt") keep priority.
 6. Reports every change it made, and finally the paths of esa-snap and gpt
    and whether a new terminal can run them without typing the path.

Usage:
    python3 snap_install.py

Only the Python standard library is used. Nothing is ever deleted except files
this script itself created (its temporary answer file and installer
symlinks). If a newer release replaces ~/esa-snap, the old copy is moved aside
to ~/esa-snap-<version>, not deleted.
"""

import os
import sys
import time
import stat
import queue
import shutil
import signal
import hashlib
import platform
import tempfile
import threading
import subprocess
import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
HOME = os.path.expanduser("~")
INSTALL_DIR = os.path.join(HOME, "esa-snap")          # where a new install goes
GUI_NAME = "esa-snap"                                  # GUI launcher, renamed
DOWNLOAD_PAGE = "https://step.esa.int/main/download/snap-download/"
INSTALLER_PREFIX = "esa-snap_all_linux-"               # Linux 64-bit, All Toolboxes
MAIN_BASE = "https://download.esa.int/step/snap/"      # main download
MIRROR_BASE = "https://step.esa.int/downloads/"        # mirror (also has checksums)
BASHRC = os.path.join(HOME, ".bashrc")
MARK_BEGIN = "# >>> ESA SNAP (added by snap_install.py) >>>"
MARK_END = "# <<< ESA SNAP (added by snap_install.py) <<<"
MIN_FREE_GB = 6                                         # rough space for an install

# Text that only SNAP's tools print (checked in the help output)
GPT_HELP_MARKERS = ["<op>|<graph-file>", "SNAP"]        # `gpt -h`
GUI_HELP_MARKERS = ["General options", "--jdkhome"]      # `snap --help` (NetBeans launcher)
VERSION_MARKER = "SNAP Release version"                  # `gpt --diag`

changes = []      # things this run changed on disk
warnings = []     # things the user should know about
errors = []       # things that failed


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def say(text=""):
    print(text)
    sys.stdout.flush()


def step(text):
    say("")
    say("== " + text)


def fmt_hms(seconds):
    seconds = int(max(0, seconds))
    return "%02d:%02d:%02d" % (seconds // 3600, (seconds // 60) % 60, seconds % 60)


def parse_version(s):
    """'14.0.0' -> (14, 0, 0); None if not a dotted number."""
    s = s.strip()
    parts = s.split(".")
    if not parts or not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def vstr(v):
    return ".".join(str(x) for x in v) if v else "unknown"


def tilde(path):
    """Show paths under the home folder as ~/... in messages."""
    if path and (path == HOME or path.startswith(HOME + os.sep)):
        return "~" + path[len(HOME):]
    return path


def ask_yes_no(question):
    """y/N prompt. Anything other than y/yes (including no input) means No."""
    try:
        ans = input(question + " [y/N] ").strip().lower()
    except EOFError:
        say("")
        say("   (no answer available: treating as No)")
        return False
    return ans in ("y", "yes")


def is_exec_file(p):
    return os.path.isfile(p) and os.access(p, os.X_OK)


def kill_group(p):
    """Stop a process and everything it started (SNAP launches a JVM)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(p.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            p.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue


def run_cmd(cmd, timeout, finish=None, echo=None):
    """Run a command and collect its output (stdout and stderr together).

    SNAP's launcher sometimes never exits after it has finished its work
    (documented for --modules commands). So `finish(line)` may return a number
    of seconds: once such a line has been seen and the output has then been
    silent for that long, the command is treated as finished and stopped.
    `echo(line)` decides whether a line is shown live.

    Returns (output_text, how) where how is 'exit:<code>', 'finished',
    'timeout' or 'error:<message>'."""
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as e:
        return "", "error:%s" % e
    q = queue.Queue()

    def reader():
        for raw in iter(p.stdout.readline, b""):
            q.put(raw.decode("utf-8", "replace"))
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    lines = []
    t0 = time.time()
    last = time.time()
    idle_needed = None
    try:
        while True:
            try:
                item = q.get(timeout=0.5)
                if item is None:
                    try:
                        rc = p.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        kill_group(p)
                        rc = None
                    return "".join(lines), "exit:%s" % rc
                lines.append(item)
                last = time.time()
                if echo is not None and echo(item):
                    say("     " + item.rstrip())
                if finish is not None:
                    n = finish(item)
                    if n is not None:
                        idle_needed = n
            except queue.Empty:
                pass
            if idle_needed is not None and time.time() - last >= idle_needed:
                kill_group(p)
                return "".join(lines), "finished"
            if time.time() - t0 > timeout:
                kill_group(p)
                return "".join(lines), "timeout"
    except KeyboardInterrupt:
        kill_group(p)
        raise


# ---------------------------------------------------------------------------
# 1. Where could SNAP be?
# ---------------------------------------------------------------------------
def candidate_roots():
    """Install folders to check (a folder whose bin/ may hold gpt and snap).

    Assembled from: ESA's installer defaults (old: ~/snap and /usr/local/snap;
    newer: ~/esa-snap and /usr/local/esa-snap, changed because of the Ubuntu
    snap conflict), common manual and packaged locations (/opt/snap,
    /opt/esa-snap, ...), the locations in wps-research misc.py find_snap(),
    versioned folder names (snap9, esa-snap-12.0, ...) directly inside the
    usual parent folders, environment variables, and whatever "gpt",
    "esa-snap" or "snap" on PATH or in the usual symlink folders points to."""
    roots = []

    def add(p):
        if not p:
            return
        p = os.path.normpath(os.path.abspath(p))
        if p not in roots:
            roots.append(p)

    names = ["esa-snap", "snap", "esa_snap", "ESA-SNAP", "SNAP", "esasnap"]
    home_parents = ["", "opt", "programs", "Programs", "software", "apps",
                    "tools", "local", ".local", "Applications"]
    sys_parents = ["/usr/local", "/opt", "/usr/share", "/usr/lib", "/usr/local/share",
                   "/opt/esa", "/opt/ESA", "/srv", "/home", "/apps", "/software", "/tools"]
    for parent in home_parents:
        for n in names:
            add(os.path.join(HOME, parent, n))
    for parent in sys_parents:
        for n in names:
            add(os.path.join(parent, n))

    # versioned / renamed folders: names starting with snap / esa-snap / esa_snap
    for parent in [HOME, os.path.join(HOME, "opt"), os.path.join(HOME, "programs"),
                   os.path.join(HOME, "software"), "/opt", "/usr/local", "/home"]:
        try:
            entries = os.listdir(parent)
        except OSError:
            continue
        for e in entries:
            low = e.lower()
            if low.startswith("snap") or low.startswith("esa-snap") or low.startswith("esa_snap"):
                add(os.path.join(parent, e))

    for var in ("SNAP_HOME", "ESA_SNAP_HOME", "SNAP_DIR"):
        if os.environ.get(var):
            add(os.environ[var])

    # follow commands on PATH and in symlink folders back to their install folder
    link_dirs = [os.path.join(HOME, "bin"), os.path.join(HOME, ".local", "bin"),
                 "/usr/local/bin", "/usr/bin", "/opt/bin"]
    paths = []
    for cmd in ("gpt", GUI_NAME, "snap"):
        w = shutil.which(cmd)
        if w:
            paths.append(w)
        for d in link_dirs:
            paths.append(os.path.join(d, cmd))
    for p in paths:
        if os.path.lexists(p):
            real = os.path.realpath(p)
            if os.path.basename(os.path.dirname(real)) == "bin":
                add(os.path.dirname(os.path.dirname(real)))
    # system folders are never a SNAP install folder (and /usr/bin/snap is
    # Ubuntu's package manager, which must not be run here)
    skip = {"/", "/usr", "/usr/local", "/opt", "/home", "/snap", "/bin", "/sbin", "/usr/share", "/usr/lib"}
    return [r for r in roots if r not in skip]


# ---------------------------------------------------------------------------
# 2. Verify candidates by their help output
# ---------------------------------------------------------------------------
def all_markers(text, markers):
    return all(m in text for m in markers)


def verify_gpt(path):
    out, how = run_cmd([path, "-h"], timeout=300,
                       finish=lambda l: 10 if "<graph-file>" in l else None)
    return all_markers(out, GPT_HELP_MARKERS)


def gpt_version(path):
    out, how = run_cmd([path, "--diag"], timeout=300,
                       finish=lambda l: 5 if VERSION_MARKER in l else None)
    for line in out.splitlines():
        i = line.find(VERSION_MARKER)
        if i >= 0:
            v = parse_version(line[i + len(VERSION_MARKER):].strip())
            if v:
                return v
    return None


def verify_gui(path):
    out, how = run_cmd([path, "--help"], timeout=300,
                       finish=lambda l: 8 if "--jdkhome" in l else None)
    return all_markers(out, GUI_HELP_MARKERS)


def inspect_root(root):
    """Return a dict describing a SNAP install folder, or None if nothing there."""
    b = os.path.join(root, "bin")
    gpt = os.path.join(b, "gpt")
    gui_new = os.path.join(b, GUI_NAME)
    gui_old = os.path.join(b, "snap")
    if not (is_exec_file(gpt) or is_exec_file(gui_new) or is_exec_file(gui_old)):
        return None
    say("   checking %s" % tilde(root))
    info = {"root": root, "gpt": None, "gui": None, "gui_old_name": False, "version": None}
    if is_exec_file(gpt):
        if verify_gpt(gpt):
            info["gpt"] = gpt
            info["version"] = gpt_version(gpt)
            say("     gpt       OK  (SNAP %s)" % vstr(info["version"]))
        else:
            say("     gpt       found, but its help output is not SNAP's: ignored")
    for gui, old in ((gui_new, False), (gui_old, True)):
        if info["gui"] is None and is_exec_file(gui):
            if verify_gui(gui):
                info["gui"] = gui
                info["gui_old_name"] = old
                say("     %-9s OK" % os.path.basename(gui))
            else:
                say("     %-9s found, but its help output is not SNAP's: ignored"
                    % os.path.basename(gui))
    if info["gpt"] is None and info["gui"] is None:
        return None
    return info


def find_installs():
    found = []
    for root in candidate_roots():
        if os.path.isdir(root):
            info = inspect_root(root)
            if info:
                found.append(info)
    return found


def pick_install(found):
    """A complete install (gpt and GUI both verified). Prefer ~/esa-snap,
    then the newest version."""
    complete = [f for f in found if f["gpt"] and f["gui"]]
    if not complete:
        return None
    complete.sort(key=lambda f: (f["root"] == INSTALL_DIR, f["version"] or (0,)), reverse=True)
    return complete[0]


# ---------------------------------------------------------------------------
# 3. Latest release, download and install
# ---------------------------------------------------------------------------
def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "snap_install.py"})
    return urllib.request.urlopen(req, timeout=timeout)


def latest_release():
    """(version tuple, installer file name, [urls]) from the SNAP download page."""
    with fetch(DOWNLOAD_PAGE) as r:
        html = r.read().decode("utf-8", "replace")
    found = {}
    i = 0
    while True:
        i = html.find(INSTALLER_PREFIX, i)
        if i < 0:
            break
        end = html.find(".sh", i)
        if end < 0:
            break
        name = html[i:end + 3]
        start = html.rfind("http", 0, i)
        url = html[start:end + 3] if start >= 0 else ""
        i = end
        v = parse_version(name[len(INSTALLER_PREFIX):-3])
        if v is None:
            continue
        lst = found.setdefault(v, [])
        bad = [c for c in ' "\'<>\n' if c in url]
        if url and not bad and url not in lst:
            lst.append(url)
    if not found:
        return None
    v = max(found)
    name = INSTALLER_PREFIX + vstr(v) + ".sh"
    mm = "%d.%d" % (v[0], v[1] if len(v) > 1 else 0)
    urls = found[v]
    for u in (MAIN_BASE + mm + "/installers/" + name, MIRROR_BASE + mm + "/installers/" + name):
        if u not in urls:
            urls.append(u)
    urls.sort(key=lambda u: 0 if u.startswith(MAIN_BASE) else 1)
    return v, name, urls


def expected_sha256(version, name):
    mm = "%d.%d" % (version[0], version[1] if len(version) > 1 else 0)
    url = MIRROR_BASE + mm + "/installers/sha256sums"
    try:
        with fetch(url) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:
        return None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].lstrip("*") == name and len(parts[0]) == 64:
            return parts[0].lower()
    return None


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url, dest):
    """Download url to dest, resuming a partial download (dest + '.part')."""
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "snap_install.py"})
    if have:
        req.add_header("Range", "bytes=%d-" % have)
        say("   resuming partial download (%.1f MB already here)" % (have / 1e6))
    try:
        r = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:          # partial file is already complete
            os.replace(part, dest)
            return
        raise
    with r:
        if have and r.status != 206:        # server ignored the resume request
            have = 0
        length = r.headers.get("Content-Length")
        total = have + int(length) if length and length.isdigit() else None
        got = have
        t0 = time.time()
        last = 0.0
        with open(part, "ab" if have else "wb") as f:
            while True:
                block = r.read(1024 * 1024)
                if not block:
                    break
                f.write(block)
                got += len(block)
                now = time.time()
                if now - last >= 0.5:
                    last = now
                    rate = (got - have) / max(1e-6, now - t0)
                    if total:
                        eta = (total - got) / rate if rate > 0 else 0
                        sys.stdout.write("\r   %5.1f%%  %.1f / %.1f MB  |  %.1f MB/s  |  ETA %s remaining   " % (
                            100.0 * got / total, got / 1e6, total / 1e6, rate / 1e6, fmt_hms(eta)))
                    else:
                        sys.stdout.write("\r   %.1f MB  |  %.1f MB/s   " % (got / 1e6, rate / 1e6))
                    sys.stdout.flush()
        sys.stdout.write("\n")
        if total and got < total:
            raise IOError("download incomplete (%d of %d bytes); re-run to resume" % (got, total))
    os.replace(part, dest)


def get_installer(release):
    """Download (or reuse) the installer in the current directory; verify it."""
    version, name, urls = release
    dest = os.path.join(os.getcwd(), name)
    sha = expected_sha256(version, name)
    if sha is None:
        warnings.append("could not get ESA's SHA-256 checksum for %s; installer not verified" % name)
    if os.path.exists(dest):
        if sha is None or sha256_of(dest) == sha:
            say("   using installer already in this folder: %s" % name)
            return dest
        say("   %s in this folder does not match ESA's checksum: downloading again" % name)
        os.remove(dest)
        if os.path.exists(dest + ".part"):
            os.remove(dest + ".part")
    last_err = None
    for url in urls:
        say("   downloading %s" % url)
        try:
            download(url, dest)
            break
        except Exception as e:
            last_err = e
            say("   failed: %s" % e)
    else:
        raise IOError("could not download the installer: %s" % last_err)
    if sha is not None:
        if sha256_of(dest) != sha:
            os.remove(dest)
            raise IOError("downloaded installer does not match ESA's SHA-256 checksum (deleted; re-run)")
        say("   SHA-256 checksum OK")
    changes.append("downloaded installer %s into %s" % (name, tilde(os.getcwd())))
    return dest


SYMLINK_DIRS = [os.path.join(HOME, "bin"), os.path.join(HOME, ".local", "bin"), "/usr/local/bin"]


def symlinks_in(dirs):
    s = set()
    for d in dirs:
        try:
            for e in os.listdir(d):
                p = os.path.join(d, e)
                if os.path.islink(p):
                    s.add(p)
        except OSError:
            pass
    return s


def run_installer(installer, target):
    """Unattended install of SNAP into `target`."""
    if os.path.exists(target) and os.listdir(target):
        raise IOError("%s already exists and is not empty; move it away and re-run" % tilde(target))
    free = shutil.disk_usage(os.path.dirname(target)).free / 1e9
    if free < MIN_FREE_GB:
        warnings.append("only %.1f GB free next to %s (SNAP needs a few GB)" % (free, tilde(target)))
    links_before = symlinks_in(SYMLINK_DIRS)
    linkdir = tempfile.mkdtemp(prefix="snap_install_links_")
    fd, varfile = tempfile.mkstemp(prefix="snap_install_", suffix=".varfile")
    # install4j answer file: user (not admin) install into `target`; any
    # command symlinks go to a throw-away folder (a "snap" symlink on PATH
    # would hide Ubuntu's snap); no Python setup; and if an older SNAP user
    # folder (~/.snap) exists, only its internal "system" part is reset
    # (graphs, auxdata etc. are kept).
    with os.fdopen(fd, "w") as f:
        f.write("sys.installationDir=%s\n" % target)
        f.write("sys.symlinkDir=%s\n" % linkdir)
        f.write("sys.adminRights$Boolean=false\n")
        f.write("sys.languageId=en\n")
        f.write("executeLauncherWithPythonAction$Boolean=false\n")
        f.write("forcePython$Boolean=false\n")
        f.write("deleteSnapDir=DESKTOP\n")
        f.write("deleteAllSnapEngineDir$Boolean=false\n")
        f.write("deleteOnlySnapDesktopDir$Boolean=true\n")
    say("   running the installer (unattended) into %s ..." % tilde(target))
    try:
        out, how = run_cmd(["sh", installer, "-q", "-varfile", varfile, "-dir", target],
                           timeout=3 * 3600, echo=lambda l: l.strip() != "")
    finally:
        os.remove(varfile)
        shutil.rmtree(linkdir, ignore_errors=True)
    if not how.startswith("exit:"):
        raise IOError("installer did not finish (%s)" % how)
    changes.append("installed SNAP into %s" % tilde(target))
    # remove any command symlinks the installer made that point into the install
    for p in sorted(symlinks_in(SYMLINK_DIRS) - links_before):
        if os.path.realpath(p).startswith(target + os.sep):
            try:
                os.remove(p)
                changes.append("removed installer-created symlink %s (avoids clashing with other tools)" % tilde(p))
            except OSError as e:
                warnings.append("installer created symlink %s; could not remove it: %s" % (tilde(p), e))
    if not is_exec_file(os.path.join(target, "bin", "gpt")):
        raise IOError("installer finished but %s/bin/gpt is missing" % tilde(target))


# ---------------------------------------------------------------------------
# 4. Rename the GUI launcher snap -> esa-snap
# ---------------------------------------------------------------------------
def userdir_of(root):
    """SNAP user folder from etc/snap.conf (default_userdir=...)."""
    conf = os.path.join(root, "etc", "snap.conf")
    try:
        with open(conf) as f:
            for line in f:
                line = line.strip()
                if line.startswith("default_userdir="):
                    v = line[len("default_userdir="):].strip().strip('"').strip("'")
                    return v.replace("${HOME}", HOME).replace("$HOME", HOME)
    except OSError:
        pass
    return None


def rename_gui(info):
    """bin/snap -> bin/esa-snap. The launcher finds its settings by its own
    name (etc/<name>.conf and etc/<name>.clusters), so links with the new
    name are added next to the originals; without them the GUI would start
    without its toolboxes and settings."""
    root = info["root"]
    old = os.path.join(root, "bin", "snap")
    new = os.path.join(root, "bin", GUI_NAME)
    etc = os.path.join(root, "etc")
    if not (os.access(os.path.join(root, "bin"), os.W_OK) and os.access(etc, os.W_OK)):
        warnings.append("GUI is %s; no write permission to rename it to %s" % (old, GUI_NAME))
        return
    made = []
    try:
        pairs = [(etc, "snap.conf", GUI_NAME + ".conf"), (etc, "snap.clusters", GUI_NAME + ".clusters")]
        ud = userdir_of(root)
        if ud:
            pairs.append((os.path.join(ud, "etc"), "snap.conf", GUI_NAME + ".conf"))
        for d, src, dst in pairs:
            if os.path.exists(os.path.join(d, src)) and not os.path.lexists(os.path.join(d, dst)):
                os.symlink(src, os.path.join(d, dst))
                made.append(os.path.join(d, dst))
        os.rename(old, new)
    except OSError as e:
        for p in made:
            os.remove(p)
        warnings.append("could not rename %s to %s: %s" % (tilde(old), GUI_NAME, e))
        return
    if verify_gui(new):
        info["gui"] = new
        info["gui_old_name"] = False
        changes.append("renamed GUI launcher %s -> %s" % (tilde(old), GUI_NAME))
        for p in made:
            changes.append("added settings link %s -> %s" % (tilde(p), os.readlink(p)))
    else:
        os.rename(new, old)
        for p in made:
            os.remove(p)
        errors.append("renamed GUI launcher did not verify; rename undone (still %s)" % tilde(old))


def fix_desktop_entries(root):
    """Menu / desktop shortcuts that still start bin/snap -> bin/esa-snap."""
    old = os.path.join(root, "bin", "snap")
    new = os.path.join(root, "bin", GUI_NAME)
    if not is_exec_file(new) or os.path.exists(old):
        return
    for d in (os.path.join(HOME, ".local", "share", "applications"), os.path.join(HOME, "Desktop")):
        try:
            entries = os.listdir(d)
        except OSError:
            continue
        for e in entries:
            if not e.endswith(".desktop"):
                continue
            p = os.path.join(d, e)
            try:
                with open(p) as f:
                    text = f.read()
            except (OSError, UnicodeDecodeError):
                continue
            new_text = text
            for after in ('"', " ", "\n", "'"):
                new_text = new_text.replace(old + after, new + after)
            if new_text.endswith(old):
                new_text = new_text[:-len(old)] + new
            if new_text != text and os.access(p, os.W_OK):
                with open(p, "w") as f:
                    f.write(new_text)
                changes.append("updated shortcut %s to start %s" % (tilde(p), GUI_NAME))


# ---------------------------------------------------------------------------
# 5. Module updates (inside the installed version)
# ---------------------------------------------------------------------------
def list_module_updates(gui):
    """[(module, installed version, new version)] using
    esa-snap --nosplash --nogui --modules --list --refresh"""
    say("   asking ESA's update centre (can take a few minutes) ...")
    out, how = run_cmd([gui, "--nosplash", "--nogui", "--modules", "--list", "--refresh"],
                       timeout=30 * 60,
                       finish=lambda l: 20 if "Code Name" in l else None,
                       echo=lambda l: l.startswith("Refreshing"))
    if "Code Name" not in out:
        raise IOError("module list not received (%s)" % how)
    ups = []
    for line in out.splitlines():
        i = line.find("Upgrade to ")
        if i >= 0:
            cols = line[:i].split()
            ups.append((cols[0] if cols else "?", cols[1] if len(cols) > 1 else "?",
                        line[i + len("Upgrade to "):].strip()))
    return ups


def update_modules(gui):
    out, how = run_cmd([gui, "--nosplash", "--nogui", "--modules", "--update-all"],
                       timeout=3 * 3600,
                       finish=lambda l: 5 if l.startswith("updates=0") else
                       (300 if l.startswith("updates=") else None),
                       echo=lambda l: l.strip() != "")
    return how


# ---------------------------------------------------------------------------
# 6. PATH in ~/.bashrc
# ---------------------------------------------------------------------------
def ensure_bashrc_path(bindir):
    shown = "$HOME" + bindir[len(HOME):] if bindir.startswith(HOME + os.sep) else bindir
    block = "\n".join([
        MARK_BEGIN,
        "# SNAP tools (esa-snap GUI, gpt). Added at the END of PATH on purpose, so",
        "# system tools with the same name (e.g. another gpt) keep priority.",
        'case ":$PATH:" in *":%s:"*) ;; *) export PATH="$PATH:%s" ;; esac' % (shown, shown),
        MARK_END,
    ])
    text = ""
    if os.path.exists(BASHRC):
        with open(BASHRC) as f:
            text = f.read()
    b = text.find(MARK_BEGIN)
    e = text.find(MARK_END)
    if b >= 0 and e > b:
        old = text[b:e + len(MARK_END)]
        if old == block:
            return
        new = text[:b] + block + text[e + len(MARK_END):]
        what = "updated SNAP PATH entry in ~/.bashrc (now %s)" % shown
    else:
        sep = "" if text == "" or text.endswith("\n") else "\n"
        new = text + sep + "\n" + block + "\n"
        what = "added %s to the end of PATH in ~/.bashrc" % shown
    if os.path.exists(BASHRC):
        backup = BASHRC + ".bak_snap_install"
        shutil.copy2(BASHRC, backup)
        changes.append("backed up ~/.bashrc to %s" % tilde(backup))
    with open(BASHRC, "w") as f:
        f.write(new)
    changes.append(what)


def other_snap_path_lines(bindir):
    """Lines in ~/.bashrc outside our block that mention a snap bin folder."""
    try:
        with open(BASHRC) as f:
            text = f.read()
    except OSError:
        return []
    b = text.find(MARK_BEGIN)
    e = text.find(MARK_END)
    if b >= 0 and e > b:
        text = text[:b] + text[e + len(MARK_END):]
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        if "PATH" in s and ("snap/bin" in s or "snap" + os.sep + "bin" in s):
            out.append(s)
    return out


def resolve_in_new_shell(names):
    """What a new interactive bash (reads ~/.bashrc) runs for each name."""
    script = "; ".join('echo "__R__%s=$(type -P %s)"' % (n, n) for n in names)
    out, how = run_cmd(["bash", "-i", "-c", script], timeout=60)
    res = {}
    for line in out.splitlines():
        i = line.find("__R__")
        if i >= 0:
            k, _, v = line[i + 5:].partition("=")
            res[k] = v.strip() or None
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def install_new(release, target):
    if os.path.exists(target) and os.listdir(target):   # check before downloading ~1 GB
        raise IOError("%s already exists and is not empty (but is not a working SNAP); "
                      "move it away and re-run" % tilde(target))
    installer = get_installer(release)
    run_installer(installer, target)
    info = inspect_root(target)
    if info is None or not info["gpt"]:
        raise IOError("new install in %s did not verify" % tilde(target))
    if info["gui"] and info["gui_old_name"]:
        rename_gui(info)
    return info


def main():
    say("snap_install.py: ESA SNAP check / install")
    say("home folder: %s   install folder for a new SNAP: %s" % (HOME, tilde(INSTALL_DIR)))
    if os.geteuid() == 0:
        warnings.append("running as root: a new install would go into root's home folder")

    step("Looking for existing SNAP installs")
    found = find_installs()
    if not found:
        say("   none found")
    chosen = pick_install(found)
    for f in found:
        if f is not chosen and not (f["gpt"] and f["gui"]):
            warnings.append("incomplete SNAP in %s (gpt %s, GUI %s)" % (
                tilde(f["root"]), "OK" if f["gpt"] else "missing", "OK" if f["gui"] else "missing"))

    step("Checking the latest SNAP release on step.esa.int")
    release = None
    try:
        release = latest_release()
        if release:
            say("   latest: SNAP %s  (%s)" % (vstr(release[0]), release[1]))
        else:
            warnings.append("could not find the Linux installer link on " + DOWNLOAD_PAGE)
    except Exception as e:
        warnings.append("could not read %s: %s" % (DOWNLOAD_PAGE, e))

    if chosen is None:
        step("No working SNAP found: installing")
        if release is None:
            errors.append("cannot install: latest release unknown (see warnings)")
        elif platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64", "amd64"):
            errors.append("cannot install: the ESA installer is for Linux 64-bit (x86_64)")
        else:
            try:
                chosen = install_new(release, INSTALL_DIR)
            except Exception as e:
                errors.append("install failed: %s" % e)
    else:
        say("")
        say("   using SNAP %s in %s" % (vstr(chosen["version"]), tilde(chosen["root"])))
        if chosen["version"] is None:
            warnings.append("could not read the installed SNAP version (gpt --diag); "
                            "newer-release check skipped")
        if chosen["gui_old_name"]:
            step("Renaming the GUI launcher snap -> %s" % GUI_NAME)
            rename_gui(chosen)
        # newer release?
        if release and chosen["version"] and release[0][:2] > chosen["version"][:2]:
            step("Newer SNAP release available")
            say("   installed: SNAP %s   available: SNAP %s" % (vstr(chosen["version"]), vstr(release[0])))
            say("   (a new release is a fresh install; module updates cannot do this)")
            if chosen["root"] == INSTALL_DIR:
                aside = INSTALL_DIR + "-" + vstr(chosen["version"])
                say("   the current %s would be moved to %s (not deleted)" % (tilde(INSTALL_DIR), tilde(aside)))
            else:
                aside = None
                say("   the new release would go into %s; %s is left as it is"
                    % (tilde(INSTALL_DIR), tilde(chosen["root"])))
            if os.path.exists(INSTALL_DIR) and chosen["root"] != INSTALL_DIR:
                say("   but %s already exists, so it cannot be installed there: skipped" % tilde(INSTALL_DIR))
            elif ask_yes_no("   Install SNAP %s now?" % vstr(release[0])):
                try:
                    if aside:
                        if os.path.exists(aside):
                            raise IOError("%s already exists" % tilde(aside))
                        os.rename(INSTALL_DIR, aside)
                        changes.append("moved previous SNAP %s to %s" % (vstr(chosen["version"]), tilde(aside)))
                    chosen = install_new(release, INSTALL_DIR)
                except Exception as e:
                    errors.append("new release install failed: %s" % e)
                    if aside and os.path.exists(aside) and not os.path.exists(INSTALL_DIR):
                        os.rename(aside, INSTALL_DIR)
                        changes.append("moved %s back to %s" % (tilde(aside), tilde(INSTALL_DIR)))
            else:
                say("   not installed")

    if chosen and chosen["gui"] and not chosen["gui_old_name"]:
        fix_desktop_entries(chosen["root"])

        step("Checking for module updates")
        try:
            ups = list_module_updates(chosen["gui"])
            if not ups:
                say("   SNAP is up to date")
            else:
                say("   %d update(s) available:" % len(ups))
                for name, have, new in ups:
                    say("     %-55s %s -> %s" % (name, have, new))
                if ask_yes_no("   Install these updates now?"):
                    how = update_modules(chosen["gui"])
                    left = list_module_updates(chosen["gui"])
                    if left:
                        errors.append("%d module update(s) still pending after updating (%s)" % (len(left), how))
                    else:
                        changes.append("installed %d SNAP module update(s)" % len(ups))
                        say("   updates installed")
                else:
                    say("   updates not installed")
        except Exception as e:
            warnings.append("could not check module updates: %s" % e)

    if chosen:
        step("PATH setup in ~/.bashrc")
        bindir = os.path.join(chosen["root"], "bin")
        ensure_bashrc_path(bindir)
        for line in other_snap_path_lines(bindir):
            warnings.append("~/.bashrc also has this older SNAP PATH line (left unchanged): %s" % line)
        if os.path.exists(os.path.join(bindir, "snap")):
            warnings.append("%s/snap still exists in a folder on PATH: `snap` may run ESA SNAP "
                            "instead of Ubuntu's snap" % tilde(bindir))

    # ------------------------------------------------------------------ report
    step("Report")
    if warnings:
        say("WARNINGS:")
        for w in warnings:
            say("  - " + w)
    if errors:
        say("ERRORS:")
        for e in errors:
            say("  - " + e)
    say("CHANGES MADE:")
    if changes:
        for c in changes:
            say("  - " + c)
    elif errors:
        say("  - none")
    else:
        say("  - none (existing install verified, nothing needed changing)")

    say("")
    if not chosen or not chosen["gpt"]:
        say("RESULT: SNAP is NOT ready (see errors above)")
        return 1
    res = resolve_in_new_shell([GUI_NAME, "gpt", "snap"])
    gui = chosen["gui"] if chosen["gui"] and not chosen["gui_old_name"] else None
    say("RESULT: ESA SNAP %s in %s" % (vstr(chosen["version"]), chosen["root"]))
    say("")
    for label, name, path in (("GUI", GUI_NAME, gui), ("CLI", "gpt", chosen["gpt"])):
        if path is None:
            if chosen["gui"]:
                say("  %s  %-9s NOT AVAILABLE: the GUI is %s (verified) but could not be renamed"
                    % (label, name, chosen["gui"]))
            else:
                say("  %s  %-9s NOT AVAILABLE" % (label, name))
            continue
        say("  %s  %-9s %s   (verified)" % (label, name, path))
        r = res.get(name)
        if r and os.path.realpath(r) == os.path.realpath(path):
            say("                 new terminals: typing `%s` runs this (PATH set in ~/.bashrc)" % name)
        elif r:
            say("                 new terminals: typing `%s` runs %s instead; use the full path above" % (name, r))
        else:
            say("                 new terminals: `%s` is NOT on PATH; use the full path above" % name)
    snap_r = res.get("snap")
    if snap_r and os.path.realpath(snap_r).startswith(chosen["root"] + os.sep):
        say("  NOTE: `snap` in new terminals runs ESA SNAP, hiding Ubuntu's snap")
    say("")
    say("  (this terminal: run `source ~/.bashrc` or open a new terminal for PATH changes)")
    return 1 if errors or gui is None else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("")
        say("INTERRUPTED. A partial download is kept and resumes on the next run.")
        sys.exit(130)
