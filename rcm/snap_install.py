#!/usr/bin/env python3
"""
snap_install.py  --  find, verify, install and update ESA SNAP

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
 3. Finds the latest SNAP release (ESA's download page; if that cannot be
    read, by asking download.esa.int directly for newer installers).
 4. If no working install is found: checks free disk space, downloads the
    latest "Linux 64-bit, All Toolboxes" installer into the current
    directory (resumable), verifies its SHA-256 checksum, and installs it
    unattended into ~/esa-snap. Then renames bin/snap to bin/esa-snap.
 5. If a newer SNAP release, or module updates, are available, asks y/N
    before installing them. Each SNAP update centre is reported as refreshed
    or failed (with the reason from SNAP's log).
 6. Adds the SNAP bin folder to the end of PATH in ~/.bashrc, so that
    system tools with the same names (e.g. another "gpt") keep priority.
 7. Reports every change it made, and finally the paths of esa-snap and gpt
    and whether a new terminal can run them without typing the path.

Certificate (SSL) problems:
    If a server's certificate cannot be verified, the script tries, in order:
    the certifi bundle (if installed), the other certificate bundles on this
    computer, and fetching missing intermediate certificates named in the
    server's certificate (the chain must still end at a root authority this
    computer already trusts; certificate checking is never switched off).
    SNAP's own Java gets the matching treatment: if an update centre refresh
    fails on a certificate, it is retried with Java allowed to fetch missing
    intermediates, then with this computer's Java certificate store. A setting
    that fixes it is also saved into SNAP's settings (snap.conf for the GUI,
    gpt.vmoptions for gpt), so gpt's own downloads (e.g. DEM tiles) work too.

Your organisation's HTTPS inspection:
    Some networks inspect HTTPS traffic and re-sign server certificates with
    their own root certificate, which neither this computer nor SNAP's Java
    trusts. The script detects this, fetches that root certificate by
    following the "CA Issuers" links in the presented certificate (it does
    not search this computer for it), shows its SHA-256 fingerprint, and
    asks y/N before trusting it. A confirmed root is
    saved in ./snap_install_files/trusted_ca.pem (re-used on later runs) and
    added to SNAP's own Java certificate store (backed up first), so SNAP's
    update centres and gpt's downloads work too. A root obtained another way
    (from IT, or exported from a browser) can be given with --ca-file.

Older SNAP installs:
    By default, verified SNAP installs with an older version than the one in
    use are deleted (they are SNAP folders this script has verified, never
    the one in use). Installs you cannot write to (e.g. system folders) are
    listed with the sudo commands that would remove them, and you are asked
    y/N before those are run. Use --keep-old to keep everything.

Memory:
    By default SNAP's Java heap (-Xmx, for both gpt and the GUI) is set to 60%
    of the RAM that is currently available, and SNAP's tile cache to half of
    that heap (the lower of the two cache sizes SNAP's own performance tool
    proposes). Settings are only changed when they differ by more than 10%.
    Use --mem-fraction to change the 60%, or --no-tune to leave memory alone.

Usage:
    python3 snap_install.py
    python3 snap_install.py --ca-file ROOT.crt      # trust this extra root certificate
    python3 snap_install.py --sha256sums SUMS_FILE  # checksum list obtained separately
    python3 snap_install.py --keep-old              # do not delete older SNAP installs
    python3 snap_install.py --mem-fraction 0.5      # heap = 50% of available RAM
    python3 snap_install.py --no-tune               # do not change memory settings
    python3 snap_install.py --no-cert-check         # see "Switching certificate checking off" below

Switching certificate checking off (--no-cert-check):
    For this script's own downloads, certificates are not checked at all.
    SNAP's Java has no such switch, so instead the exact certificates that
    ESA's site presents right now are added to SNAP's trusted list. That
    trusts that one site's current certificate only (it cannot vouch for
    any other site); if the presented certificate changes, re-run with the
    flag. Anything on the network between you and ESA could then feed this
    script and SNAP altered files without being noticed. Prefer --ca-file.

Files this script creates are kept in the current directory: the installer,
and the folder ./snap_install_files (confirmed certificates; its temporary
installer files are removed after use). Nothing goes to a system temp folder.

Only the Python standard library is used. Nothing is ever deleted except files
this script itself created (its temporary answer file and installer
symlinks). If a newer release replaces ~/esa-snap, the old copy is moved aside
to ~/esa-snap-<version>, not deleted. Files it edits are backed up first
(<file>.bak_snap_install).
"""

import os
import sys
import ssl
import time
import queue
import shutil
import signal
import socket
import hashlib
import platform
import argparse
import threading
import subprocess
import urllib.parse
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
USER_AGENT = "snap_install.py"

# Text that only SNAP's tools print (checked in the help output)
GPT_HELP_MARKERS = ["<op>|<graph-file>", "SNAP"]        # `gpt -h`
GUI_HELP_MARKERS = ["General options", "--jdkhome"]      # `snap --help` (NetBeans launcher)
VERSION_MARKER = "SNAP Release version"                  # `gpt --diag`

# Certificate bundles that may exist on Linux systems (tried if the default fails)
CA_FILES = ["/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt",
            "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem", "/etc/ssl/ca-bundle.pem",
            "/etc/ssl/cert.pem", "/usr/local/share/certs/ca-root-nss.crt"]
# This computer's Java certificate store (kept current by the OS), if present
JAVA_TRUSTSTORES = ["/etc/ssl/certs/java/cacerts", "/etc/pki/java/cacerts",
                    "/etc/pki/ca-trust/extracted/java/cacerts"]
JAVA_AIA_FLAG = "-J-Dcom.sun.security.enableAIAcaIssuers=true"
PERSIST_MARK = "# added by snap_install.py (certificate fix)"
WORK_DIR = os.path.join(os.getcwd(), "snap_install_files")   # all script data, in the current folder
TRUSTED_CA = os.path.join(WORK_DIR, "trusted_ca.pem")
# The order in which find_snap() in wps-research py/misc.py looks for gpt
FIND_SNAP_ORDER = ["/usr/local/snap", "/usr/local/esa-snap", "/opt/snap",
                   os.path.join(HOME, "snap"), os.path.join(HOME, "esa-snap")]
NOISE = ("HeadlessException", "NullPointerException", "InvocationTargetException",
         "EventDispatchThread", "java.awt.")
RELEVANT = ("PKIX", "Reading URL", "SSLHandshake", "certification path", "UnknownHost",
            "ConnectException", "SocketTimeout", "timed out", "407", "NoRouteToHost")

changes = []      # things this run changed on disk
warnings = []     # things the user should know about
errors = []       # things that failed
notes = []        # extra explanations shown in the report
centre_report = []   # (update centre, status) for the report
update_summary = [None]  # one line about module updates for the report
refresh_failed = [False]  # update centres could not be refreshed
ARGS = argparse.Namespace(ca_file=None, sha256sums=None, keep_old=False, mem_fraction=0.6, no_tune=False,
                          no_cert_check=False)
TUNE_MARK = "# memory set by snap_install.py"


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


def rel(path):
    """Paths in the current folder are shown relative to it."""
    r = os.path.relpath(path)
    return r if not r.startswith("..") else tilde(path)


def is_exec_file(p):
    return os.path.isfile(p) and os.access(p, os.X_OK)


def backup(path):
    b = path + ".bak_snap_install"
    if not os.path.exists(b):
        shutil.copy2(path, b)
        changes.append("backed up %s to %s" % (tilde(path), tilde(b)))


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
# Network: HTTPS with certificate fallbacks, retries, proxies
# ---------------------------------------------------------------------------
class TLSFailure(Exception):
    """A server's certificate could not be verified by any method."""


_tls = {}          # host -> (ssl context, how it was verified)


def is_cert_error(x):
    return isinstance(x, ssl.SSLCertVerificationError) or (
        isinstance(x, ssl.SSLError) and "CERTIFICATE_VERIFY_FAILED" in str(x))


def describe_error(e):
    """Readable reason for urllib / ssl / socket errors."""
    if isinstance(e, urllib.error.HTTPError):
        return "HTTP %d %s" % (e.code, e.reason)
    if isinstance(e, urllib.error.URLError):
        return describe_error(e.reason) if isinstance(e.reason, BaseException) else str(e.reason)
    if isinstance(e, ssl.SSLCertVerificationError) and getattr(e, "verify_message", None):
        return "certificate verify failed: %s" % e.verify_message
    return str(e)


# --- tiny DER reader (certificate names and AIA links; no external modules) --
def der_tlv(b, i):
    """(tag, content start, content end) of the DER element at b[i]."""
    tag = b[i]
    ln = b[i + 1]
    j = i + 2
    if ln & 0x80:
        n = ln & 0x7f
        ln = int.from_bytes(b[j:j + n], "big")
        j += n
    if j + ln > len(b):
        raise ValueError("truncated DER")
    return tag, j, j + ln


def cert_subject_issuer(der):
    """Raw DER subject and issuer names of an X.509 certificate."""
    tag, s, e = der_tlv(der, 0)                 # Certificate
    if tag != 0x30:
        raise ValueError("not a certificate")
    tag, s, e = der_tlv(der, s)                 # tbsCertificate
    if tag != 0x30:
        raise ValueError("not a certificate")
    i = s
    tag, cs, ce = der_tlv(der, i)
    if tag == 0xA0:                             # [0] version
        i = ce
    i = der_tlv(der, i)[2]                      # serialNumber
    i = der_tlv(der, i)[2]                      # signature algorithm
    tag, cs, ce = der_tlv(der, i)               # issuer
    issuer = der[i:ce]
    i = ce
    i = der_tlv(der, i)[2]                      # validity
    tag, cs, ce = der_tlv(der, i)               # subject
    subject = der[i:ce]
    if tag != 0x30:
        raise ValueError("not a certificate")
    return subject, issuer


def name_text(name_der):
    """Common name (or organisation) from a DER name."""
    for oid in (b"\x06\x03\x55\x04\x03", b"\x06\x03\x55\x04\x0a"):   # CN, O
        k = name_der.find(oid)
        if k >= 0:
            try:
                tag, cs, ce = der_tlv(name_der, k + len(oid))
                return name_der[cs:ce].decode("utf-8", "replace")
            except (ValueError, IndexError):
                pass
    return "?"


def ca_issuer_urls(der):
    """'CA Issuers' links (Authority Information Access) in a certificate."""
    oid = b"\x06\x08\x2b\x06\x01\x05\x05\x07\x30\x02"   # 1.3.6.1.5.5.7.48.2
    urls = []
    i = 0
    while True:
        i = der.find(oid, i)
        if i < 0:
            break
        j = i + len(oid)
        if j < len(der) and der[j] == 0x86:              # URI
            try:
                tag, cs, ce = der_tlv(der, j)
                urls.append(der[cs:ce].decode("ascii", "replace"))
            except (ValueError, IndexError):
                pass
        i = j
    return urls


def certs_from_bytes(data):
    """DER certificates from a downloaded .crt/.cer/.pem (PKCS#7 is skipped)."""
    out = []
    if b"-----BEGIN CERTIFICATE-----" in data:
        text = data.decode("ascii", "replace")
        parts = text.split("-----BEGIN CERTIFICATE-----")[1:]
        for p in parts:
            body = p.split("-----END CERTIFICATE-----")[0]
            try:
                out.append(ssl.PEM_cert_to_DER_cert(
                    "-----BEGIN CERTIFICATE-----" + body + "-----END CERTIFICATE-----\n"))
            except ValueError:
                pass
    else:
        try:
            cert_subject_issuer(data)
            out.append(data)
        except (ValueError, IndexError):
            pass
    return out


def proxy_for(scheme, host):
    p = urllib.request.getproxies().get(scheme)
    if not p:
        return None
    try:
        if urllib.request.proxy_bypass(host):
            return None
    except Exception:
        pass
    return p


def raw_connect(host, port=443, timeout=20):
    """TCP connection to host, through an HTTP(S)_PROXY tunnel if one is set."""
    proxy = proxy_for("https", host)
    if not proxy:
        return socket.create_connection((host, port), timeout=timeout)
    u = urllib.parse.urlsplit(proxy if "://" in proxy else "http://" + proxy)
    s = socket.create_connection((u.hostname, u.port or 8080), timeout=timeout)
    req = "CONNECT %s:%d HTTP/1.1\r\nHost: %s:%d\r\n" % (host, port, host, port)
    if u.username:
        import base64
        cred = "%s:%s" % (urllib.parse.unquote(u.username), urllib.parse.unquote(u.password or ""))
        req += "Proxy-Authorization: Basic %s\r\n" % base64.b64encode(cred.encode()).decode()
    s.sendall((req + "\r\n").encode())
    resp = b""
    while b"\r\n\r\n" not in resp and len(resp) < 65536:
        chunk = s.recv(4096)
        if not chunk:
            break
        resp += chunk
    first = resp.split(b"\r\n", 1)[0].decode("latin-1")
    if " 200" not in first:
        s.close()
        raise OSError("proxy refused tunnel: " + first)
    return s


def server_chain(host, port=443):
    """Certificates the server sends (not verified), as DER."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    s = raw_connect(host, port)
    try:
        with ctx.wrap_socket(s, server_hostname=host) as t:
            if hasattr(t, "get_unverified_chain"):
                chain = t.get_unverified_chain()
                if chain:
                    return list(chain)
            leaf = t.getpeercert(binary_form=True)
            return [leaf] if leaf else []
    finally:
        s.close()


def strict_context(cadata=None, cafile=None):
    """Verifying context whose chain must end at a trusted ROOT certificate."""
    ctx = ssl.create_default_context(cafile=cafile)
    if hasattr(ssl, "VERIFY_X509_PARTIAL_CHAIN"):
        ctx.verify_flags &= ~ssl.VERIFY_X509_PARTIAL_CHAIN
    if cadata:
        ctx.load_verify_locations(cadata=cadata)
    return ctx


def tls_works(ctx, host, port=443):
    """True if the certificate of https://host:port/ verifies with ctx."""
    req = urllib.request.Request("https://%s:%d/" % (host, port), method="HEAD",
                                 headers={"User-Agent": USER_AGENT})
    try:
        urllib.request.urlopen(req, timeout=30, context=ctx).close()
        return True
    except urllib.error.HTTPError:
        return True                      # an HTTP answer means TLS was fine
    except urllib.error.URLError as e:
        if is_cert_error(e.reason):
            return False
        raise


def intermediates_context(host, port=443):
    """Fetch intermediate certificates the server failed to send, using the
    'CA Issuers' links inside its certificates. Self-signed (root)
    certificates are never added, and partial chains are not accepted, so the
    chain must still end at a root this computer already trusts."""
    try:
        todo = server_chain(host, port)
    except Exception:
        return None
    pems = []
    seen = set()
    for depth in range(4):
        urls = []
        for d in todo:
            for u in ca_issuer_urls(d):
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
        if not urls:
            return None
        todo = []
        for u in urls:
            try:
                req = urllib.request.Request(u, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read(1024 * 1024)
            except Exception:
                continue
            for der in certs_from_bytes(data):
                try:
                    subj, iss = cert_subject_issuer(der)
                except (ValueError, IndexError):
                    continue
                if subj == iss:
                    continue             # a root: must already be trusted locally
                pems.append(ssl.DER_cert_to_PEM_cert(der))
                todo.append(der)
        if not todo:
            return None
        ctx = strict_context(cadata="".join(pems))
        try:
            if tls_works(ctx, host, port):
                return ctx
        except Exception:
            return None
    return None


def diagnose_chain(host, port=443):
    """One line describing the certificate chain the server sends."""
    try:
        chain = server_chain(host, port)
    except Exception as e:
        return "could not read the server's certificates: %s" % e
    parts = []
    for der in chain:
        try:
            subj, iss = cert_subject_issuer(der)
            parts.append("'%s' (issued by '%s')" % (name_text(subj), name_text(iss)))
        except (ValueError, IndexError):
            parts.append("(unreadable certificate)")
    return "server sends: " + " <- ".join(parts) if parts else "server sent no certificates"


# --- extra root certificates the user trusts (e.g. organisation's HTTPS inspection)
extra_ca = {"pems": [], "source": None, "declined": False, "untrusted_root": None, "no_link": False}


def fingerprint(der):
    h = hashlib.sha256(der).hexdigest().upper()
    return ":".join(h[i:i + 2] for i in range(0, len(h), 2))


def load_extra_ca(ca_file):
    """Certificates from --ca-file, or those confirmed on an earlier run
    (./snap_install_files/trusted_ca.pem)."""
    if ca_file:
        path, source = ca_file, "--ca-file " + ca_file
    elif os.path.isfile(TRUSTED_CA):
        path, source = TRUSTED_CA, rel(TRUSTED_CA) + " (confirmed on an earlier run)"
    else:
        return
    try:
        with open(path, "rb") as f:
            ders = certs_from_bytes(f.read())
    except OSError as e:
        raise SystemExit("ERROR: cannot read %s: %s" % (path, e))
    if not ders:
        raise SystemExit("ERROR: no certificate found in %s" % path)
    extra_ca["pems"] = [ssl.DER_cert_to_PEM_cert(d) for d in ders]
    extra_ca["source"] = source
    for d in ders:
        try:
            subj, iss = cert_subject_issuer(d)
            say("trusting extra certificate '%s' (SHA-256 %s) from %s" % (name_text(subj), fingerprint(d), source))
        except (ValueError, IndexError):
            pass


def base_context():
    """Default verification plus any extra trusted certificates."""
    if not extra_ca["pems"]:
        return None
    ctx = ssl.create_default_context()
    ctx.load_verify_locations(cadata="".join(extra_ca["pems"]))
    return ctx


def fetch_issuer_chain(host, port):
    """(leaf, [issuer certificates], root or None): the certificates the
    server sends, plus issuers fetched from their 'CA Issuers' links, up to
    a self-signed root."""
    chain = server_chain(host, port)
    if not chain:
        return None, [], None
    leaf = chain[0]
    issuers = list(chain[1:])
    cur = issuers[-1] if issuers else leaf
    for _ in range(6):
        subj, iss = cert_subject_issuer(cur)
        if subj == iss:
            return leaf, issuers, cur
        nxt = None
        for u in ca_issuer_urls(cur):
            try:
                req = urllib.request.Request(u, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=30) as r:
                    ders = certs_from_bytes(r.read(1024 * 1024))
                if ders:
                    nxt = ders[0]
                    break
            except Exception:
                continue
        if nxt is None:
            return leaf, issuers, None
        issuers.append(nxt)
        cur = nxt
    return leaf, issuers, None


def offer_org_ca(host, port):
    """The certificate chain ends at a root this computer does not trust
    (typically an organisation's HTTPS inspection). Fetch that root, show
    its fingerprint, and ask the user before trusting it."""
    try:
        leaf, issuers, root = fetch_issuer_chain(host, port)
    except Exception as e:
        say("   could not read the certificates of %s: %s" % (host, e))
        return None
    if leaf is None:
        return None
    lsubj, liss = cert_subject_issuer(leaf)
    if root is None:
        top = cert_subject_issuer(issuers[-1])[1] if issuers else liss
        extra_ca["untrusted_root"] = name_text(top)
        say("   the certificate for %s is issued by '%s', which this computer does not trust," % (host, name_text(top)))
        say("   and it names no download link for that issuer, so it cannot be fetched automatically.")
        extra_ca["no_link"] = True
        return None
    rname = name_text(cert_subject_issuer(root)[0])
    extra_ca["untrusted_root"] = rname
    os.makedirs(WORK_DIR, exist_ok=True)
    say("")
    say("   The certificate presented for %s is issued by '%s'" % (host, name_text(liss)))
    say("   and its chain ends at the root certificate '%s', which this computer does not trust." % rname)
    say("   ESA's own certificate comes from a public certificate authority, so this usually means")
    say("   your organisation's network inspects HTTPS traffic and re-signs certificates")
    say("   (less likely: this computer's list of trusted certificates is out of date).")
    say("   Certificates (copies saved in %s):" % rel(WORK_DIR))
    for k, der in enumerate(issuers):
        s_, i_ = cert_subject_issuer(der)
        path = os.path.join(WORK_DIR, "ca_chain_%d.crt" % (k + 1))
        with open(path, "w") as f:
            f.write(ssl.DER_cert_to_PEM_cert(der))
        say("     %s  '%s' (issued by '%s')%s" % (rel(path), name_text(s_), name_text(i_),
                                                  "  <- ROOT" if der == root else ""))
    say("   ROOT SHA-256 fingerprint:")
    say("     " + fingerprint(root))
    say("   Before trusting it, compare this fingerprint with one from your IT department, or with")
    say("   the root certificate a browser on an organisation-managed computer shows for https://%s." % host)
    if not ask_yes_no("   Trust this root certificate for this script and for SNAP?"):
        extra_ca["declined"] = True
        say("   not trusted")
        return None
    pems = [ssl.DER_cert_to_PEM_cert(d) for d in issuers]
    with open(TRUSTED_CA, "w") as f:
        f.write("".join(pems))
    extra_ca["pems"] = pems
    extra_ca["source"] = rel(TRUSTED_CA) + " (confirmed now)"
    changes.append("saved the root certificate you confirmed ('%s', SHA-256 %s) and its intermediates to %s"
                   % (rname, fingerprint(root), rel(TRUSTED_CA)))
    ctx = base_context()
    try:
        if tls_works(ctx, host, port):
            return ctx
    except Exception:
        pass
    warnings.append("the root certificate you confirmed still does not verify %s" % host)
    return None


def find_tls_context(host, port, first_error):
    """Try other ways to verify host's certificate. Returns a context or
    raises TLSFailure. Never turns verification off."""
    say("   certificate of %s could not be verified (%s); trying other certificate sources ..."
        % (host, describe_error(first_error)))
    alts = []
    try:
        import certifi
        alts.append(("the certifi certificate bundle", certifi.where()))
    except ImportError:
        pass
    for f in CA_FILES:
        if os.path.isfile(f):
            alts.append(("certificate bundle " + f, f))
    for how, f in alts:
        try:
            ctx = strict_context(cafile=f)
            if tls_works(ctx, host, port):
                return ctx, how
        except (ssl.SSLError, OSError, urllib.error.URLError):
            continue
    ctx = intermediates_context(host, port)
    if ctx is not None:
        return ctx, "missing intermediate certificates fetched from the CA (chain verified to a trusted root)"
    if not extra_ca["declined"] and not extra_ca["pems"]:
        ctx = offer_org_ca(host, port)
        if ctx is not None:
            return ctx, "CONFIRMED"
    elif extra_ca["untrusted_root"] is None:
        try:
            leaf, issuers, root = fetch_issuer_chain(host, port)
            if root is not None:
                extra_ca["untrusted_root"] = name_text(cert_subject_issuer(root)[0])
        except Exception:
            pass
    raise TLSFailure("certificate of %s cannot be verified (%s); %s" % (
        host, describe_error(first_error), diagnose_chain(host, port)))


def open_url(url, method=None, headers=None, timeout=60, tries=3):
    """urlopen with: proxies from the environment, retries on network errors
    and server errors (5xx), and certificate fallbacks for HTTPS."""
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname
    port = parts.port or 443
    https = parts.scheme == "https"
    key = "%s:%d" % (host, port)
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    attempt = 0
    while True:
        attempt += 1
        if https and ARGS.no_cert_check:
            ctx = ssl._create_unverified_context()          # --no-cert-check: no certificate checking
        else:
            ctx = _tls[key][0] if (https and key in _tls) else (base_context() if https else None)
        req = urllib.request.Request(url, headers=h, method=method)
        try:
            return urllib.request.urlopen(req, timeout=timeout, context=ctx)
        except urllib.error.HTTPError as e:
            if e.code >= 500 and attempt < tries:
                say("   server error %d from %s; retrying in %d s" % (e.code, host, 5 * attempt))
                time.sleep(5 * attempt)
                continue
            raise
        except urllib.error.URLError as e:
            if https and is_cert_error(e.reason) and key not in _tls:
                ctx, how = find_tls_context(host, port, e.reason)       # raises TLSFailure
                _tls[key] = (ctx, how)
                if how == "CONFIRMED":
                    say("   certificate of %s now verified with the root certificate you confirmed" % host)
                else:
                    warnings.append("certificate of %s verified only using %s; this computer's list of "
                                    "trusted certificates may be out of date" % (host, how))
                    say("   certificate of %s verified using %s" % (host, how))
                attempt -= 1
                continue
            if is_cert_error(e.reason) or attempt >= tries:
                raise
            say("   network problem reaching %s (%s); retrying in %d s" % (host, describe_error(e), 5 * attempt))
            time.sleep(5 * attempt)
        except (socket.timeout, ConnectionError) as e:
            if attempt >= tries:
                raise
            say("   network problem reaching %s (%s); retrying in %d s" % (host, e, 5 * attempt))
            time.sleep(5 * attempt)


def cert_help():
    if extra_ca["untrusted_root"] and extra_ca["no_link"]:
        notes.append("The certificates presented for ESA's site are issued by '%s', which this computer does not "
                     "trust; most likely your organisation's network inspects HTTPS traffic. That certificate "
                     "names no download link for its issuer, so get the issuer's root certificate yourself: "
                     "ask IT for it, or on an organisation-managed computer open https://step.esa.int in "
                     "Chrome/Edge, click the icon left of the address -> 'Connection is secure' -> 'Certificate "
                     "is valid' -> 'Details', select the TOP certificate in the hierarchy and 'Export' it "
                     "(Base64/PEM; in Firefox: padlock -> 'Connection secure' -> 'More information' -> 'View "
                     "Certificate' -> its last tab -> 'PEM (cert)'). Copy that file here and run "
                     "`python3 snap_install.py --ca-file <file>`; the script then also adds it to SNAP's Java, so "
                     "SNAP's update centres and gpt's own downloads (e.g. DEM tiles) work." % extra_ca["untrusted_root"])
        return
    if extra_ca["untrusted_root"]:
        notes.append("The certificates presented for ESA's site end at '%s', which this computer does not "
                     "trust; most likely your organisation's network inspects HTTPS traffic. Fix: re-run and "
                     "answer y when the script offers to trust that root (after checking its fingerprint), or get "
                     "the root certificate from IT (or export it from a browser on a managed computer) and run "
                     "`python3 snap_install.py --ca-file <file>`. The script then also adds it to SNAP's Java, "
                     "so SNAP's update centres and gpt's own downloads (e.g. DEM tiles) work."
                     % extra_ca["untrusted_root"])
        return
    notes.append("Certificate problems usually mean this computer's list of trusted certificate "
                 "authorities is out of date. To fix: Ubuntu/Debian: "
                 "`sudo apt-get update && sudo apt-get install --only-upgrade ca-certificates`; "
                 "RHEL/Rocky/Fedora: `sudo dnf update ca-certificates`; or for this script only: "
                 "`python3 -m pip install --upgrade certifi` (it is used automatically). Then re-run.")


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


def install_of(path, found):
    """The found install that a (resolved) path belongs to, if any."""
    real = os.path.realpath(path)
    for f in found:
        if real.startswith(os.path.realpath(f["root"]) + os.sep):
            return f
    return None


# ---------------------------------------------------------------------------
# 3. Latest release, download and install
# ---------------------------------------------------------------------------
def installer_name(v):
    return INSTALLER_PREFIX + vstr(v) + ".sh"


def installer_urls(v):
    mm = "%d.%d" % (v[0], v[1] if len(v) > 1 else 0)
    name = installer_name(v)
    return [MAIN_BASE + mm + "/installers/" + name, MIRROR_BASE + mm + "/installers/" + name]


def latest_from_page():
    """(version tuple, installer file name, [urls]) from the SNAP download page."""
    with open_url(DOWNLOAD_PAGE) as r:
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
    urls = found[v]
    for u in installer_urls(v):
        if u not in urls:
            urls.append(u)
    urls.sort(key=lambda u: 0 if u.startswith(MAIN_BASE) else 1)
    return v, installer_name(v), urls


def url_exists(url):
    try:
        with open_url(url, method="HEAD", tries=2):
            return True
    except urllib.error.HTTPError as e:
        if e.code in (403, 404, 410):
            return False
        if e.code not in (405, 501):          # HEAD not allowed: ask for 1 byte instead
            raise
    try:
        with open_url(url, headers={"Range": "bytes=0-0"}, tries=2):
            return True
    except urllib.error.HTTPError as e:
        if e.code in (403, 404, 410):
            return False
        raise


def latest_by_probing(installed):
    """Fallback when the download page cannot be read: ask the download
    server directly which newer installers exist (x.0.0 releases)."""
    if installed:
        majors = list(range(installed[0], installed[0] + 5))
    else:
        majors = list(range(10, 21))
    best = None
    for m in majors:
        v = (m, 0, 0)
        if installed and v[:2] <= installed[:2]:
            continue
        if url_exists(installer_urls(v)[0]):
            best = v
    if best is None:
        return None
    return best, installer_name(best), installer_urls(best)


def latest_release(installed):
    try:
        rel = latest_from_page()
        if rel:
            say("   latest release (from ESA's download page): SNAP %s" % vstr(rel[0]))
            return rel
        warnings.append("installer link not found on " + DOWNLOAD_PAGE)
        say("   the download page did not contain the Linux installer link")
    except TLSFailure as e:
        say("   could not read the download page: %s" % e)
        warnings.append(str(e))
        cert_help()
    except Exception as e:
        say("   could not read the download page: %s" % describe_error(e))
        warnings.append("could not read %s: %s" % (DOWNLOAD_PAGE, describe_error(e)))
    say("   asking download.esa.int directly for newer installers ...")
    try:
        rel = latest_by_probing(installed)
    except TLSFailure as e:
        say("   failed: %s" % e)
        warnings.append(str(e))
        cert_help()
        return None
    except Exception as e:
        say("   failed: %s" % describe_error(e))
        warnings.append("could not check download.esa.int: %s" % describe_error(e))
        return None
    if rel:
        say("   newer release found on download.esa.int: SNAP %s" % vstr(rel[0]))
    elif installed:
        say("   no release newer than %s found on download.esa.int" % vstr(installed))
    else:
        say("   no installer found on download.esa.int")
    return rel


def expected_sha256(version, name):
    mm = "%d.%d" % (version[0], version[1] if len(version) > 1 else 0)
    url = MIRROR_BASE + mm + "/installers/sha256sums"
    try:
        with open_url(url) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception as e:
        say("   could not get ESA's checksum list: %s" % describe_error(e))
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


def remote_size(url):
    try:
        with open_url(url, method="HEAD", tries=2) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n and n.isdigit() else None
    except Exception:
        return None


def download(url, dest):
    """Download url to dest, resuming a partial download (dest + '.part').
    A dropped connection is retried (resuming) up to 5 times."""
    part = dest + ".part"
    for attempt in range(1, 6):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"Range": "bytes=%d-" % have} if have else None
        if have:
            say("   resuming partial download (%.1f MB already here)" % (have / 1e6))
        try:
            r = open_url(url, headers=headers, timeout=180)
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:          # partial file is already complete
                os.replace(part, dest)
                return
            raise
        try:
            with r:
                if have and r.status != 206:    # server ignored the resume request
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
                    raise IOError("connection dropped at %d of %d bytes" % (got, total))
            os.replace(part, dest)
            return
        except (IOError, OSError, socket.timeout) as e:
            if attempt == 5:
                raise IOError("download incomplete: %s (re-run to resume)" % e)
            say("   download interrupted (%s); resuming in %d s" % (e, 5 * attempt))
            time.sleep(5 * attempt)


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


def get_installer(release, target):
    """Download (or reuse) the installer in the current directory; verify it.
    Checks free disk space before downloading anything."""
    version, name, urls = release
    dest = os.path.join(os.getcwd(), name)
    sha = expected_sha256(version, name)
    if sha is None and ARGS.sha256sums:
        sha = sha256_from_file(ARGS.sha256sums, name)
    if os.path.exists(dest):
        if sha is not None and sha256_of(dest) == sha:
            say("   using installer already in this folder: %s (checksum OK)" % name)
            return dest
        if sha is None:
            say("   using installer already in this folder: %s" % name)
        else:
            say("   %s in this folder does not match ESA's checksum: downloading again" % name)
            os.remove(dest)
            if os.path.exists(dest + ".part"):
                os.remove(dest + ".part")
    if not os.path.exists(dest):
        # disk space, before downloading ~1 GB
        size = remote_size(urls[0])
        have = os.path.getsize(dest + ".part") if os.path.exists(dest + ".part") else 0
        free_here = shutil.disk_usage(os.getcwd()).free
        if size and free_here < (size - have) * 1.05 + 50e6:
            raise IOError("not enough space in this folder for the installer (%.1f GB needed, %.1f GB free)"
                          % ((size - have) / 1e9, free_here / 1e9))
        free_target = shutil.disk_usage(os.path.dirname(target)).free / 1e9
        if free_target < MIN_FREE_GB:
            raise IOError("only %.1f GB free for %s; SNAP needs about %d GB"
                          % (free_target, tilde(target), MIN_FREE_GB))
        say("   free space OK (%.1f GB here, %.1f GB for %s)" % (free_here / 1e9, free_target, tilde(target)))
        last_err = None
        for url in urls:
            say("   downloading %s" % url)
            try:
                download(url, dest)
                break
            except TLSFailure as e:
                last_err = e
                cert_help()
                say("   failed: %s" % e)
            except Exception as e:
                last_err = e
                say("   failed: %s" % describe_error(e))
        else:
            raise IOError("could not download the installer: %s" % last_err)
        changes.append("downloaded installer %s into %s" % (name, tilde(os.getcwd())))
    if sha is not None:
        if sha256_of(dest) != sha:
            os.remove(dest)
            raise IOError("downloaded installer does not match ESA's SHA-256 checksum (deleted; re-run)")
        say("   SHA-256 checksum OK")
    else:
        mine = sha256_of(dest)
        mm = "%d.%d" % (version[0], version[1] if len(version) > 1 else 0)
        sums_url = MIRROR_BASE + mm + "/installers/sha256sums"
        say("   ESA's checksum for %s could not be obtained, so the installer cannot be verified" % name)
        say("   its SHA-256 is: %s" % mine)
        say("   (compare with the line for %s in %s, e.g. in a browser;" % (name, sums_url))
        say("   or save that list and re-run with --sha256sums <file>)")
        if not ask_yes_no("   Install it anyway?"):
            raise IOError("installer not verified; not installed (re-run when ESA's site is reachable, "
                          "or with --sha256sums)")
        warnings.append("installer %s was installed without checksum verification" % name)
        notes.append("To verify the installer later: its SHA-256 is %s; compare it with the line for %s in %s."
                     % (mine, name, sums_url))
    return dest


def sha256_from_file(path, name):
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[-1].lstrip("*") == name and len(parts[0]) == 64:
                    say("   checksum for %s taken from %s" % (name, path))
                    return parts[0].lower()
    except OSError as e:
        warnings.append("could not read --sha256sums file %s: %s" % (path, e))
        return None
    warnings.append("%s has no line for %s" % (path, name))
    return None


def run_installer(installer, target):
    """Unattended install of SNAP into `target`."""
    if os.path.exists(target) and os.listdir(target):
        raise IOError("%s already exists and is not empty; move it away and re-run" % tilde(target))
    links_before = symlinks_in(SYMLINK_DIRS)
    os.makedirs(WORK_DIR, exist_ok=True)
    linkdir = os.path.join(WORK_DIR, "installer_symlinks")
    shutil.rmtree(linkdir, ignore_errors=True)
    os.makedirs(linkdir)
    varfile = os.path.join(WORK_DIR, "install.varfile")
    # install4j answer file: user (not admin) install into `target`; any
    # command symlinks go to a throw-away folder (a "snap" symlink on PATH
    # would hide Ubuntu's snap); no Python setup; and if an older SNAP user
    # folder (~/.snap) exists, only its internal "system" part is reset
    # (graphs, auxdata etc. are kept).
    with open(varfile, "w") as f:
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
        try:
            os.rmdir(WORK_DIR)          # only if nothing else (e.g. certificates) is kept there
        except OSError:
            pass
    if not how.startswith("exit:"):
        raise IOError("installer did not finish (%s)" % how)
    if how != "exit:0":
        warnings.append("installer ended with %s" % how)
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
        say("   not renamed: no write permission in %s" % tilde(root))
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
        say("   renamed %s -> %s (settings links added; GUI verified)" % (tilde(old), GUI_NAME))
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
def proxy_java_flags():
    """Give SNAP's Java the proxy from http(s)_proxy (Java ignores those)."""
    flags = []
    proxies = urllib.request.getproxies()
    for scheme in ("http", "https"):
        p = proxies.get(scheme)
        if not p:
            continue
        u = urllib.parse.urlsplit(p if "://" in p else "http://" + p)
        if u.hostname:
            flags += ["-J-D%s.proxyHost=%s" % (scheme, u.hostname),
                      "-J-D%s.proxyPort=%d" % (scheme, u.port or 8080)]
    np = proxies.get("no")
    if flags and np:
        hosts = []
        for h in np.split(","):
            h = h.strip()
            if h:
                hosts.append("*" + h if h.startswith(".") else h)
        flags.append("-J-Dhttp.nonProxyHosts=" + "|".join(hosts))
    return flags


def java_flag_sets():
    """Ways to run SNAP's Java, tried in order when a refresh fails on a certificate."""
    sets = [("as installed", [])]
    sets.append(("Java allowed to fetch missing intermediate certificates", [JAVA_AIA_FLAG]))
    for ts in JAVA_TRUSTSTORES:
        if os.path.isfile(ts):
            sets.append(("Java using this computer's certificate store (%s)" % ts,
                         [JAVA_AIA_FLAG, "-J-Djavax.net.ssl.trustStore=" + ts,
                          "-J-Djavax.net.ssl.trustStorePassword=changeit"]))
            break
    return sets


def snap_log_lines(root, since):
    """Error lines written to SNAP's log (messages.log) since `since`."""
    ud = userdir_of(root) or os.path.join(HOME, ".snap", "system")
    log = os.path.join(ud, "var", "log", "messages.log")
    try:
        if os.path.getmtime(log) < since - 5:
            return []
        with open(log, errors="replace") as f:
            lines = f.readlines()[-600:]
    except OSError:
        return []
    out = []
    for line in lines:
        s = line.strip()
        if relevant_line(s) and s not in out:
            out.append(s[:220])
    return out[:4]


def relevant_line(s):
    """Lines that explain a failed refresh (not headless-mode noise)."""
    if any(n in s for n in NOISE):
        return False
    return any(k in s for k in RELEVANT)


def classify(text):
    t = text
    if ("PKIX" in t or "certification path" in t or "SSLHandshake" in t or "CertificateException" in t
            or "ValidatorException" in t or "certificate" in t.lower()):
        return "certificate"
    if "UnknownHost" in t:
        return "dns"
    if "407" in t and "roxy" in t:
        return "proxy"
    if "ConnectException" in t or "timed out" in t or "SocketTimeout" in t or "NoRouteToHost" in t:
        return "network"
    if "404" in t or "FileNotFound" in t:
        return "missing"
    return "unknown"


KIND_TEXT = {"certificate": "certificate not trusted by SNAP's Java",
             "dns": "server name could not be resolved (no network/DNS?)",
             "proxy": "proxy needs a login",
             "network": "could not connect (network or proxy)",
             "missing": "update centre address not found on the server",
             "unknown": "reason unknown"}


def parse_upgrades(out):
    ups = []
    for line in out.splitlines():
        i = line.find("Upgrade to ")
        if i >= 0:
            cols = line[:i].split()
            ups.append((cols[0] if cols else "?", cols[1] if len(cols) > 1 else "?",
                        line[i + len("Upgrade to "):].strip()))
    return ups


def list_module_updates(info, quiet=False, record=True):
    """Refresh every SNAP update centre and list module updates.
    Returns (updates list or None, java flags that worked). Each update
    centre is reported as refreshed or FAILED with the reason."""
    gui = info["gui"]
    root = info["root"]
    base = proxy_java_flags()
    sets = java_flag_sets()
    centre_status = {}
    order = []
    if record:
        refresh_failed[0] = False
    for n, (label, flags) in enumerate(sets):
        if n > 0:
            say("   retrying with %s ..." % label)
        elif not quiet:
            say("   refreshing SNAP's update centres (can take a few minutes) ...")
        t0 = time.time()
        out, how = run_cmd([gui] + base + flags + ["--nosplash", "--nogui", "--modules", "--list", "--refresh"],
                           timeout=30 * 60,
                           finish=lambda l: 20 if "Code Name" in l else None,
                           echo=None)
        centres = [l[len("Refreshing "):].strip() for l in out.splitlines() if l.startswith("Refreshing ")]
        for c in centres:
            if c not in order:
                order.append(c)
        if "Code Name" in out:
            for c in centres:
                prev = centre_status.get(c, "")
                if n == 0:
                    centre_status[c] = "refreshed"
                elif prev.startswith("FAILED"):
                    centre_status[c] = "refreshed after a retry with %s (first attempt failed: %s)" % (
                        label, prev[len("FAILED: "):])
                else:
                    centre_status[c] = "refreshed (with %s)" % label
                if not quiet:
                    say("     update centre %-32s refreshed" % ("'" + c + "':"))
            if record:
                for c in order:
                    centre_report.append((c, centre_status.get(c, "not refreshed")))
            return parse_upgrades(out), flags
        # failed: the last centre being refreshed is the one that failed
        reasons = snap_log_lines(root, t0)
        extra = [l.strip()[:220] for l in out.splitlines() if relevant_line(l.strip())]
        kind = classify(out + "\n" + "\n".join(reasons))
        for c in centres[:-1]:
            if not centre_status.get(c, "").startswith("refreshed"):
                centre_status[c] = "refreshed"
            say("     update centre %-32s refreshed" % ("'" + c + "':"))
        if centres:
            centre_status[centres[-1]] = "FAILED: %s" % KIND_TEXT[kind]
            say("     update centre %-32s FAILED (%s, %s)" % ("'" + centres[-1] + "':", KIND_TEXT[kind], how))
        else:
            say("     refresh FAILED before any update centre was contacted (%s)" % how)
        for l in (reasons or extra)[:4]:
            say("       " + l)
        if kind not in ("certificate", "unknown"):
            break
        if kind == "certificate" and extra_ca["untrusted_root"]:
            if extra_ca["pems"]:
                say("   not retrying: SNAP's Java still rejects the certificate even with the trusted root added")
            else:
                say("   not retrying: the certificate ends at '%s', which is not trusted; other Java"
                    % extra_ca["untrusted_root"])
                say("   settings cannot fix that (see the note in the report)")
                cert_help()
            break
    if record:
        for c in order:
            centre_report.append((c, centre_status.get(c, "not refreshed")))
    # fall back to what SNAP already knows (cached catalogues)
    say("   update centres could not all be refreshed; listing from SNAP's cached catalogues instead")
    out, how = run_cmd([gui] + base + ["--nosplash", "--nogui", "--modules", "--list"],
                       timeout=15 * 60, finish=lambda l: 10 if "Code Name" in l else None)
    if record:
        refresh_failed[0] = True
    if "Code Name" in out:
        return parse_upgrades(out), None
    return None, None


UPDATE_BATCH = 20        # modules per update batch (each batch is downloaded AND installed)
LIST_NAME_WIDTH = 50     # SNAP's module list cuts longer names to this many characters
UPDATE_NOISE = ("DISPLAY", "OpenJDK 64-Bit Server VM warning", "WARNING:",
                "Will update ", "updates=")


def update_modules(gui, flags, ups):
    """Install module updates in batches. SNAP downloads all requested
    updates before installing any of them, so one big run loses everything
    if it is interrupted; batches keep each finished batch, and a re-run of
    this script then offers only the updates that are still missing.
    SNAP prints a module's name again for every bit of download progress;
    each name is shown once here."""
    names = [u[0] for u in ups]               # as SNAP's list shows them (possibly cut short)
    batches = [names[i:i + UPDATE_BATCH] for i in range(0, len(names), UPDATE_BATCH)]
    say("   installing %d update(s) in %d batch(es) of up to %d; each batch is installed before the next"
        % (len(names), len(batches), UPDATE_BATCH))
    say("   (if this is interrupted, re-run the script: finished batches are kept, and only the")
    say("   remaining updates are offered. After each batch SNAP may sit silent for up to 2 minutes;")
    say("   it is then stopped, because SNAP's updater often does not exit by itself.)")
    hows = []
    t0 = time.time()
    for k, batch in enumerate(batches, 1):
        say("   batch %d/%d (%d modules):" % (k, len(batches), len(batch)))
        seen = set()

        def echo(line, seen=seen):
            t = "".join(c for c in line if c.isprintable()).strip()
            if not any(c.isalnum() for c in t):           # blank / control-character-only progress lines
                return False
            if t.startswith("Nothing to update"):         # SNAP repeats every pattern here; say it briefly
                if "nothing" not in seen:
                    seen.add("nothing")
                    say("     (SNAP: nothing left to update for this batch)")
                return False
            if any(t.startswith(n) or n in t for n in UPDATE_NOISE) or t in seen:
                return False
            seen.add(t)
            return True

        # a name cut short in SNAP's list is matched by its first characters
        patterns = [n + ".*" if len(n) >= LIST_NAME_WIDTH else n for n in batch]
        out, how = run_cmd([gui] + proxy_java_flags() + (flags or []) +
                           ["--nosplash", "--nogui", "--modules", "--update"] + patterns,
                           timeout=2 * 3600,
                           finish=lambda l: 5 if l.startswith("updates=0") else
                           (120 if l.startswith("updates=") else None),
                           echo=echo)
        # count from SNAP's own module list: batch modules no longer offered an update.
        # (This also catches updates SNAP applied at start-up, e.g. ones staged by an
        # earlier interrupted run, which the update command itself reports as 0.)
        pending = pending_upgrades(gui, flags)
        if pending is None:
            done = "?"
        else:
            done = str(sum(1 for n in batch if n not in pending))
        say("   batch %d/%d finished: %s of %d module(s) now up to date (%s elapsed)"
            % (k, len(batches), done, len(batch), fmt_hms(time.time() - t0)))
        hows.append(how)
    return ", ".join(sorted(set(hows)))


def pending_upgrades(gui, flags):
    """Names SNAP's module list (from its saved catalogue, no refresh) still
    marks 'Upgrade to ...', or None if the list could not be read."""
    out, how = run_cmd([gui] + proxy_java_flags() + (flags or []) +
                       ["--nosplash", "--nogui", "--modules", "--list"],
                       timeout=15 * 60, finish=lambda l: 10 if "Code Name" in l else None)
    if "Code Name" not in out:
        return None
    return set(u[0] for u in parse_upgrades(out))


def persist_java_flags(info, jflags):
    """Save a Java setting that fixed the certificate problem into SNAP's
    settings: etc/snap.conf (GUI, also used as esa-snap.conf via its link)
    and bin/gpt.vmoptions (gpt). Each change is verified and undone if SNAP
    no longer starts."""
    root = info["root"]
    plain = [f[2:] for f in jflags]                  # -J-Dx=y -> -Dx=y
    conf = os.path.join(root, "etc", "snap.conf")
    vm = os.path.join(root, "bin", "gpt.vmoptions")
    manual = "add %s to default_options in %s and, one per line, %s to %s" % (
        " ".join(jflags), conf, " ".join(plain), vm)
    if not (os.access(conf, os.W_OK) and (not os.path.exists(vm) or os.access(vm, os.W_OK))):
        warnings.append("SNAP needs a certificate setting but its files are not writable; to fix by hand: " + manual)
        return
    # GUI: snap.conf default_options="..."
    with open(conf) as f:
        old_conf = f.read()
    lines = old_conf.splitlines(True)
    edited = False
    for k, line in enumerate(lines):
        if line.lstrip().startswith("default_options="):
            q2 = line.rfind('"')
            q1 = line.find('"')
            if q2 > q1 >= 0:
                add = [fl for fl in jflags if fl not in line]
                if add:
                    lines[k] = line[:q2] + " " + " ".join(add) + line[q2:]
                    edited = True
            break
    if edited:
        backup(conf)
        with open(conf, "w") as f:
            f.write("".join(lines))
        if info["gui"] and verify_gui(info["gui"]):
            changes.append("added %s to default_options in %s (GUI)" % (" ".join(jflags), tilde(conf)))
        else:
            with open(conf, "w") as f:
                f.write(old_conf)
            warnings.append("GUI did not start with the certificate setting; %s left unchanged" % tilde(conf))
    # gpt: gpt.vmoptions, one option per line
    old_vm = ""
    if os.path.exists(vm):
        with open(vm) as f:
            old_vm = f.read()
    have = [l.strip() for l in old_vm.splitlines()]
    add = [p for p in plain if p not in have]
    if add:
        if os.path.exists(vm):
            backup(vm)
        sep = "" if old_vm == "" or old_vm.endswith("\n") else "\n"
        with open(vm, "w") as f:
            f.write(old_vm + sep + PERSIST_MARK + "\n" + "\n".join(add) + "\n")
        if verify_gpt(info["gpt"]):
            changes.append("added %s to %s (gpt)" % (" ".join(add), tilde(vm)))
        else:
            with open(vm, "w") as f:
                f.write(old_vm)
            warnings.append("gpt did not start with the certificate setting; %s left unchanged" % tilde(vm))


def import_ca_into_java(info, pems=None, source=None):
    """Add certificates to SNAP's own Java (jre/lib/security/cacerts): by
    default the extra trusted root certificates."""
    if pems is None:
        pems, source = extra_ca["pems"], extra_ca["source"]
    if not pems:
        return
    root = info["root"]
    keytool = os.path.join(root, "jre", "bin", "keytool")
    cacerts = os.path.join(root, "jre", "lib", "security", "cacerts")
    src = TRUSTED_CA if os.path.isfile(TRUSTED_CA) else (ARGS.ca_file or "<certificate file>")
    manual = "keytool -importcert -noprompt -alias snap-install-ca -file %s -keystore %s -storepass changeit" % (
        src, cacerts)
    if not os.path.isfile(cacerts):
        warnings.append("SNAP's own Java certificate store not found (%s); trusted certificate not added to it"
                        % tilde(cacerts))
        return
    if not is_exec_file(keytool):
        keytool = shutil.which("keytool")
        if not keytool:
            warnings.append("no keytool found to add the trusted certificate to SNAP's Java; run: " + manual)
            return
    if not os.access(cacerts, os.W_OK):
        warnings.append("SNAP's Java certificate store is not writable; to add the certificate run: " + manual)
        return
    os.makedirs(WORK_DIR, exist_ok=True)
    added = 0
    for pem in pems:
        der = ssl.PEM_cert_to_DER_cert(pem)
        alias = "snap-install-" + hashlib.sha256(der).hexdigest()[:16]
        out, how = run_cmd([keytool, "-list", "-alias", alias, "-keystore", cacerts,
                            "-storepass", "changeit"], timeout=120)
        if how == "exit:0":
            continue                        # already there
        path = os.path.join(WORK_DIR, alias + ".crt")
        with open(path, "w") as f:
            f.write(pem)
        if added == 0:
            backup(cacerts)
        out, how = run_cmd([keytool, "-importcert", "-noprompt", "-alias", alias, "-file", path,
                            "-keystore", cacerts, "-storepass", "changeit"], timeout=120)
        if how == "exit:0":
            added += 1
        else:
            warnings.append("could not add certificate %s to SNAP's Java: %s" % (alias, out.strip()[-200:]))
    if added:
        say("   added %d trusted certificate(s) to SNAP's Java" % added)
        changes.append("added %d trusted certificate(s) from %s to SNAP's Java (%s)"
                       % (added, source, tilde(cacerts)))


def pin_presented_certs(info):
    """--no-cert-check for SNAP's Java: trust the exact certificates that
    ESA's site presents now (the hosts in this script's settings)."""
    hosts = []
    for u in (DOWNLOAD_PAGE, MIRROR_BASE, MAIN_BASE):
        parts = urllib.parse.urlsplit(u)
        if parts.scheme == "https" and (parts.hostname, parts.port or 443) not in hosts:
            hosts.append((parts.hostname, parts.port or 443))
    pems = []
    for host, port in hosts:
        try:
            chain = server_chain(host, port)
        except Exception as e:
            warnings.append("could not read the certificate presented by %s: %s" % (host, e))
            continue
        for der in chain:
            pem = ssl.DER_cert_to_PEM_cert(der)
            if pem not in pems:
                pems.append(pem)
        try:
            subj, iss = cert_subject_issuer(chain[0])
            say("   %s presents '%s' (issued by '%s'), SHA-256 %s"
                % (host, name_text(subj), name_text(iss), fingerprint(chain[0])))
        except (ValueError, IndexError):
            pass
    if pems:
        import_ca_into_java(info, pems, "the certificates ESA's site presents now (--no-cert-check)")


def warn_find_snap_order(chosen, found):
    """find_snap() in wps-research py/misc.py takes the first gpt in a fixed order."""
    for r in FIND_SNAP_ORDER:
        if os.path.isfile(os.path.join(r, "bin", "gpt")):
            if os.path.realpath(r) != os.path.realpath(chosen["root"]):
                other = install_of(os.path.join(r, "bin", "gpt"), found)
                ver = vstr(other["version"]) if other else "unknown version"
                warnings.append("scripts using find_snap() (wps-research py/misc.py) will run %s/bin/gpt "
                                "(SNAP %s), not %s, because find_snap() checks %s first; retire that install "
                                "or move the %s check first in find_snap()"
                                % (tilde(r), ver, tilde(chosen["root"]), tilde(r), tilde(chosen["root"])))
            return


def safe_to_delete(root, chosen):
    """Only a verified SNAP folder that is not, and does not contain, the one in use."""
    r = os.path.realpath(root)
    c = os.path.realpath(chosen["root"])
    if r == c or c.startswith(r + os.sep) or r.startswith(c + os.sep):
        return False
    if r in ("/", os.path.realpath(HOME)) or len([x for x in r.split(os.sep) if x]) < 2:
        return False
    if not os.path.isfile(os.path.join(r, "bin", "gpt")):
        return False
    etc = os.path.join(r, "etc")
    return os.path.isfile(os.path.join(etc, "snap.conf")) or os.path.isfile(os.path.join(etc, "snap.clusters"))


def remove_old_installs(chosen, found):
    """Delete verified SNAP installs older than the one in use (default)."""
    if not chosen.get("version"):
        say("   installed version unknown: older installs not deleted")
        return
    old = [f for f in found if f["root"] != chosen["root"] and f.get("version")
           and f["version"] < chosen["version"] and safe_to_delete(f["root"], chosen)]
    if not old:
        say("   no older SNAP installs found")
        return
    if ARGS.keep_old:
        for f in old:
            say("   kept SNAP %s in %s (--keep-old)" % (vstr(f["version"]), tilde(f["root"])))
        return
    link_dirs = SYMLINK_DIRS + ["/usr/bin", "/opt/bin"]
    sudo_cmds = []
    gui_new = chosen["gui"] if chosen.get("gui") and not chosen.get("gui_old_name") else None
    for f in old:
        root = os.path.realpath(f["root"])
        # links (e.g. ~/bin/esa-snap, /usr/local/bin/gpt) that point into this install
        links = [l for l in symlinks_in(link_dirs) if os.path.realpath(l).startswith(root + os.sep)]
        parent_ok = os.access(os.path.dirname(root), os.W_OK) and os.access(root, os.W_OK)
        if parent_ok:
            try:
                shutil.rmtree(root)
                say("   deleted SNAP %s in %s" % (vstr(f["version"]), tilde(root)))
                changes.append("deleted older SNAP %s in %s" % (vstr(f["version"]), tilde(root)))
            except OSError as e:
                warnings.append("could not fully delete %s: %s" % (tilde(root), e))
                parent_ok = False
        if not parent_ok:
            say("   SNAP %s in %s: no permission to delete it" % (vstr(f["version"]), tilde(root)))
            sudo_cmds.append(["rm", "-rf", root])
        for l in links:
            name = os.path.basename(l)
            if name == GUI_NAME and gui_new:
                cmd = ["ln", "-sf", gui_new, l]          # keep `esa-snap` working: point it at the new one
                what = "repointed %s to %s" % (tilde(l), tilde(gui_new))
            else:
                cmd = ["rm", "-f", l]                     # e.g. a gpt/snap link that would clash with other tools
                what = "removed link %s (pointed into the deleted install)" % tilde(l)
            if os.access(os.path.dirname(l), os.W_OK):
                try:
                    if cmd[0] == "ln":
                        os.remove(l)
                        os.symlink(gui_new, l)
                    else:
                        os.remove(l)
                    changes.append(what)
                except OSError as e:
                    warnings.append("could not change link %s: %s" % (tilde(l), e))
            else:
                sudo_cmds.append(cmd)
    if not sudo_cmds:
        return
    say("   These need administrator rights (sudo):")
    for c in sudo_cmds:
        say("     sudo " + " ".join(c))
    if not ask_yes_no("   Run them now (sudo may ask for your password)?"):
        for c in sudo_cmds:
            warnings.append("not done (needs sudo): sudo " + " ".join(c))
        return
    for c in sudo_cmds:
        r = subprocess.run(["sudo"] + c)            # interactive: sudo may prompt for a password
        if r.returncode == 0:
            changes.append("ran: sudo " + " ".join(c))
        else:
            errors.append("failed (exit %d): sudo %s" % (r.returncode, " ".join(c)))


def available_ram_bytes():
    """RAM currently available for new programs (Linux MemAvailable)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return None


def size_to_mb(text):
    """'64G' / '65536m' / '1024' (MB) -> MB, or None."""
    t = text.strip().lower()
    mult = 1.0
    if t and t[-1] in "kmgt":
        mult = {"k": 1.0 / 1024, "m": 1.0, "g": 1024.0, "t": 1024.0 * 1024}[t[-1]]
        t = t[:-1]
    try:
        return float(t) * mult
    except ValueError:
        return None


def differs(old_mb, new_mb):
    return old_mb is None or abs(old_mb - new_mb) > 0.10 * new_mb


def tune_memory(info):
    """Heap (-Xmx) for gpt and the GUI = fraction of available RAM; tile cache = half the heap."""
    avail = available_ram_bytes()
    if not avail:
        warnings.append("could not read the available RAM; memory settings left unchanged")
        return
    frac = ARGS.mem_fraction
    heap_gb = max(2, int(avail * frac / 1024 ** 3))
    heap_mb = heap_gb * 1024
    cache_mb = int(heap_mb * 0.5)
    say("   available RAM now: %.1f GB -> SNAP heap %d GB (%d%%), tile cache %.1f GB (half the heap)"
        % (avail / 1024 ** 3, heap_gb, int(round(frac * 100)), cache_mb / 1024.0))
    root = info["root"]

    # gpt: bin/gpt.vmoptions, one option per line
    vm = os.path.join(root, "bin", "gpt.vmoptions")
    if info.get("gpt"):
        old = open(vm).read() if os.path.exists(vm) else ""
        lines = old.splitlines()
        cur = [l.strip() for l in lines if l.strip().startswith("-Xmx")]
        cur_mb = size_to_mb(cur[-1][4:]) if cur else None
        if differs(cur_mb, heap_mb):
            if os.path.exists(vm) and not os.access(vm, os.W_OK):
                warnings.append("%s not writable; gpt heap not changed" % tilde(vm))
            else:
                new_lines = [l for l in lines if not l.strip().startswith("-Xmx") and l.strip() != TUNE_MARK]
                new_lines += [TUNE_MARK, "-Xmx%dG" % heap_gb]
                if os.path.exists(vm):
                    backup(vm)
                with open(vm, "w") as f:
                    f.write("\n".join(new_lines) + "\n")
                if verify_gpt(info["gpt"]):
                    changes.append("gpt heap %s -> %d GB in %s" % (
                        ("%.0f GB" % (cur_mb / 1024)) if cur_mb else "default", heap_gb, tilde(vm)))
                else:
                    with open(vm, "w") as f:
                        f.write(old)
                    warnings.append("gpt did not start with -Xmx%dG; %s left unchanged" % (heap_gb, tilde(vm)))
        else:
            say("   gpt heap already about %d GB" % heap_gb)

    # GUI: etc/snap.conf default_options="... -J-Xmx..."
    conf = os.path.join(root, "etc", "snap.conf")
    gui = info["gui"] if info.get("gui") and not info.get("gui_old_name") else None
    if gui and os.path.isfile(conf):
        old = open(conf).read()
        lines = old.splitlines(True)
        for k, line in enumerate(lines):
            if line.lstrip().startswith("default_options="):
                q1, q2 = line.find('"'), line.rfind('"')
                if q2 <= q1:
                    break
                opts = line[q1 + 1:q2].split()
                cur = [o for o in opts if o.startswith("-J-Xmx")]
                cur_mb = size_to_mb(cur[-1][6:]) if cur else None
                if not differs(cur_mb, heap_mb):
                    say("   GUI heap already about %d GB" % heap_gb)
                    break
                if not os.access(conf, os.W_OK):
                    warnings.append("%s not writable; GUI heap not changed" % tilde(conf))
                    break
                opts = [o for o in opts if not o.startswith("-J-Xmx")] + ["-J-Xmx%dG" % heap_gb]
                lines[k] = line[:q1 + 1] + " ".join(opts) + line[q2:]
                backup(conf)
                with open(conf, "w") as f:
                    f.write("".join(lines))
                if verify_gui(gui):
                    changes.append("GUI heap %s -> %d GB in %s" % (
                        ("%.0f GB" % (cur_mb / 1024)) if cur_mb else "default", heap_gb, tilde(conf)))
                else:
                    with open(conf, "w") as f:
                        f.write(old)
                    warnings.append("GUI did not start with -J-Xmx%dG; %s left unchanged" % (heap_gb, tilde(conf)))
                break

    # tile cache: ~/.snap/etc/snap.properties (snap.jai.tileCacheSize, in MB), used by gpt and the GUI
    props = os.path.join(HOME, ".snap", "etc", "snap.properties")
    key = "snap.jai.tileCacheSize"
    old = open(props).read() if os.path.exists(props) else ""
    lines = old.splitlines()
    cur_mb = None
    for l in lines:
        if l.split("=", 1)[0].strip() == key and "=" in l:
            cur_mb = size_to_mb(l.split("=", 1)[1])
    if not differs(cur_mb, cache_mb):
        say("   tile cache already about %.1f GB" % (cache_mb / 1024.0))
        return
    new_lines = [l for l in lines if l.split("=", 1)[0].strip() != key and l.strip() != TUNE_MARK]
    new_lines += [TUNE_MARK, "%s = %d" % (key, cache_mb)]
    try:
        os.makedirs(os.path.dirname(props), exist_ok=True)
        if os.path.exists(props):
            backup(props)
        with open(props, "w") as f:
            f.write("\n".join(new_lines) + "\n")
        changes.append("tile cache %s -> %.1f GB (%s = %d MB) in %s" % (
            ("%.1f GB" % (cur_mb / 1024)) if cur_mb else "default", cache_mb / 1024.0, key, cache_mb, tilde(props)))
    except OSError as e:
        warnings.append("could not set the tile cache in %s: %s" % (tilde(props), e))


def offer_delete_installer(info):
    p = info.get("installer")
    if not p or not os.path.isfile(p):
        return
    size = os.path.getsize(p) / 1e9
    if ask_yes_no("   Delete the installer %s (%.1f GB) from this folder? It is not needed after installing."
                  % (os.path.basename(p), size)):
        os.remove(p)
        changes.append("deleted installer %s (your choice)" % os.path.basename(p))


def check_module_updates(info):
    lock = os.path.join(userdir_of(info["root"]) or os.path.join(HOME, ".snap", "system"), "lock")
    if os.path.exists(lock):
        warnings.append("SNAP's lock file %s exists: if the SNAP GUI is open, close it and re-run "
                        "for reliable updates" % tilde(lock))
    ups, flags = list_module_updates(info)
    if flags:
        persist_java_flags(info, flags)
    if ups is None:
        warnings.append("could not check module updates (see update centre lines above)")
        update_summary[0] = "module updates: could not be checked"
        return
    if not ups and refresh_failed[0]:
        say("   SNAP's saved catalogue lists no updates, but it could not be refreshed, so this says nothing")
        update_summary[0] = "module updates: UNKNOWN (update centres could not be refreshed)"
        return
    if not ups:
        say("   no module updates available")
        update_summary[0] = "module updates: none available"
        return
    if refresh_failed[0]:
        say("   (from SNAP's saved catalogue; the update centres could not be refreshed)")
    say("   %d module update(s) available:" % len(ups))
    for name, have, new in ups:
        say("     %-55s %s -> %s" % (name, have, new))
    if not ask_yes_no("   Install these updates now?"):
        say("   updates not installed")
        update_summary[0] = "module updates: %d available, not installed (your choice)" % len(ups)
        return
    how = update_modules(info["gui"], flags, ups)
    say("   checking the result ...")
    left, _ = list_module_updates(info, quiet=True, record=False)
    if left is None:
        warnings.append("updates were run (%s) but the result could not be checked" % how)
        update_summary[0] = "module updates: %d installed, result not confirmed" % len(ups)
    elif left:
        errors.append("%d of %d module update(s) still pending after updating (%s)" % (len(left), len(ups), how))
        update_summary[0] = "module updates: %d still pending" % len(left)
    else:
        changes.append("installed %d SNAP module update(s)" % len(ups))
        say("   all updates installed")
        update_summary[0] = "module updates: %d installed, none left" % len(ups)


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
            say("   ~/.bashrc already adds %s to PATH" % shown)
            return
        new = text[:b] + block + text[e + len(MARK_END):]
        what = "updated SNAP PATH entry in ~/.bashrc (now %s)" % shown
    else:
        sep = "" if text == "" or text.endswith("\n") else "\n"
        new = text + sep + "\n" + block + "\n"
        what = "added %s to the end of PATH in ~/.bashrc" % shown
    if os.path.exists(BASHRC):
        backup(BASHRC)
    with open(BASHRC, "w") as f:
        f.write(new)
    changes.append(what)
    say("   " + what)


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
    installer = get_installer(release, target)
    run_installer(installer, target)
    info = inspect_root(target)
    if info is None or not info["gpt"]:
        raise IOError("new install in %s did not verify" % tilde(target))
    if info["gui"] and info["gui_old_name"]:
        rename_gui(info)
    info["installer"] = installer
    return info


def main(argv=None):
    try:
        return _main(argv)
    except KeyboardInterrupt:
        say("")
        say("INTERRUPTED. Re-run to continue: a partial download resumes, finished update batches are")
        say("kept and only the remaining updates are offered.")
        return 130


def _main(argv=None):
    ap = argparse.ArgumentParser(description="Find, verify, install and update ESA SNAP (gpt + esa-snap GUI).")
    ap.add_argument("--ca-file", help="extra root certificate to trust (e.g. your organisation's "
                                      "HTTPS-inspection root), PEM or DER")
    ap.add_argument("--sha256sums", help="ESA's sha256sums list for the installer, obtained separately")
    ap.add_argument("--keep-old", action="store_true", help="do not delete older SNAP installs")
    ap.add_argument("--mem-fraction", type=float, default=0.6,
                    help="fraction of currently available RAM for SNAP's heap (default 0.6)")
    ap.add_argument("--no-tune", action="store_true", help="do not change SNAP's memory settings")
    ap.add_argument("--no-cert-check", action="store_true",
                    help="do not check certificates (this script), and trust the certificates ESA's site "
                         "presents now (SNAP's Java); see the top of this file")
    a = ap.parse_args(argv)
    if not 0.05 <= a.mem_fraction <= 0.95:
        ap.error("--mem-fraction must be between 0.05 and 0.95")
    ARGS.ca_file = a.ca_file
    ARGS.sha256sums = a.sha256sums
    ARGS.keep_old = a.keep_old
    ARGS.mem_fraction = a.mem_fraction
    ARGS.no_tune = a.no_tune
    ARGS.no_cert_check = a.no_cert_check
    say("snap_install.py: ESA SNAP check / install")
    say("home folder: %s   install folder for a new SNAP: %s" % (HOME, tilde(INSTALL_DIR)))
    if os.geteuid() == 0:
        warnings.append("running as root: a new install would go into root's home folder")
    load_extra_ca(ARGS.ca_file)
    if ARGS.no_cert_check:
        say("")
        say("WARNING: --no-cert-check: certificates are NOT checked for this script's downloads, and SNAP's")
        say("Java will trust the certificates ESA's site presents now. Altered files on the network would")
        say("not be noticed.")
        warnings.append("certificate checking was OFF for this run (--no-cert-check); installer checksums "
                        "were also fetched without it")

    step("Looking for existing SNAP installs")
    found = find_installs()
    chosen = pick_install(found)
    complete = [f for f in found if f["gpt"] and f["gui"]]
    if not found:
        say("   none found")
    else:
        say("")
        for f in found:
            kind = "complete" if (f["gpt"] and f["gui"]) else "incomplete"
            mark = "  <- using this one" if f is chosen else ""
            say("   %-28s SNAP %-8s %s%s" % (tilde(f["root"]), vstr(f["version"]), kind, mark))
        if chosen and len(complete) > 1:
            why = "preferred location" if chosen["root"] == INSTALL_DIR else "newest version"
            say("   (%d complete installs; using %s: %s)" % (len(complete), tilde(chosen["root"]), why))
    for f in found:
        if f is not chosen and not (f["gpt"] and f["gui"]):
            warnings.append("incomplete SNAP in %s (gpt %s, GUI %s)" % (
                tilde(f["root"]), "OK" if f["gpt"] else "missing", "OK" if f["gui"] else "missing"))

    step("Checking the latest SNAP release")
    release = latest_release(chosen["version"] if chosen else None)

    if chosen is None:
        step("No working SNAP found: installing")
        if release is None:
            errors.append("cannot install: the latest release could not be determined (see warnings)")
        elif platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64", "amd64"):
            errors.append("cannot install: the ESA installer is for Linux 64-bit (x86_64)")
        else:
            try:
                chosen = install_new(release, INSTALL_DIR)
                offer_delete_installer(chosen)
            except Exception as e:
                errors.append("install failed: %s" % describe_error(e))
    else:
        say("   installed: SNAP %s in %s" % (vstr(chosen["version"]), tilde(chosen["root"])))
        if chosen["version"] is None:
            warnings.append("could not read the installed SNAP version (gpt --diag); "
                            "newer-release check skipped")
        if chosen["gui_old_name"]:
            step("Renaming the GUI launcher snap -> %s" % GUI_NAME)
            rename_gui(chosen)
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
                        prev = dict(chosen)
                        prev["root"] = aside
                        found.append(prev)
                    chosen = install_new(release, INSTALL_DIR)
                    offer_delete_installer(chosen)
                except Exception as e:
                    errors.append("new release install failed: %s" % describe_error(e))
                    if aside and os.path.exists(aside) and not os.path.exists(INSTALL_DIR):
                        os.rename(aside, INSTALL_DIR)
                        changes.append("moved %s back to %s" % (tilde(aside), tilde(INSTALL_DIR)))
            else:
                say("   not installed (your choice)")
        elif release and chosen["version"]:
            say("   SNAP %s is the latest release" % vstr(chosen["version"]))

    if chosen and extra_ca["pems"]:
        step("Trusted certificate for SNAP's Java")
        import_ca_into_java(chosen)
    if chosen and ARGS.no_cert_check:
        step("Trusting ESA's presented certificates in SNAP's Java (--no-cert-check)")
        pin_presented_certs(chosen)

    if chosen and chosen["gui"] and not chosen["gui_old_name"]:
        fix_desktop_entries(chosen["root"])
        step("Checking for module updates")
        try:
            check_module_updates(chosen)
        except Exception as e:
            warnings.append("could not check module updates: %s" % describe_error(e))
            update_summary[0] = "module updates: could not be checked"

    if chosen:
        step("Older SNAP installs")
        remove_old_installs(chosen, found)
        if not ARGS.no_tune and chosen.get("gpt"):
            step("Memory settings")
            tune_memory(chosen)
        warn_find_snap_order(chosen, found)
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
    if notes:
        say("NOTES:")
        for n in dict.fromkeys(notes):
            say("  - " + n)
    if centre_report or update_summary[0]:
        say("UPDATE CENTRES:")
        for c, s in centre_report:
            say("  - %-34s %s" % ("'" + c + "':", s))
        if update_summary[0]:
            say("  - " + update_summary[0])
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
            real = os.path.realpath(r)
            other = install_of(r, found)
            what = r if real == r else "%s -> %s" % (r, real)
            if other:
                what += " (the SNAP %s install in %s)" % (vstr(other["version"]), tilde(other["root"]))
            elif name == "gpt":
                what += " (not SNAP's gpt)"
            say("                 new terminals: typing `%s` runs %s" % (name, what))
            say("                 instead of this one, because %s comes earlier in PATH."
                % os.path.dirname(r))
            if name == GUI_NAME and os.path.islink(r):
                sudo = "" if os.access(os.path.dirname(r), os.W_OK) else "sudo "
                say("                 to use this one: %sln -sf %s %s" % (sudo, path, r))
            else:
                say("                 use the full path above to run this one")
        else:
            say("                 new terminals: `%s` is NOT on PATH; use the full path above" % name)
    snap_r = res.get("snap")
    if snap_r and install_of(snap_r, found):
        say("  NOTE: `snap` in new terminals runs ESA SNAP (%s), hiding Ubuntu's snap" % snap_r)
    say("")
    say("  (this terminal: run `source ~/.bashrc` or open a new terminal for PATH changes)")
    return 1 if errors or gui is None else 0


if __name__ == "__main__":
    sys.exit(main())

