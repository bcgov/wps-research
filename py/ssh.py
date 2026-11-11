#!/usr/bin/env python3
"""20261111
sshctl.py: start, stop, restart or inspect the OpenSSH server on Ubuntu/Debian.

    sshctl.py               show status (the default; changes nothing)
    sshctl.py --start       install if missing, start now, enable at boot
    sshctl.py --stop        stop now, disable at boot (package stays installed)
    sshctl.py --restart     check the config, then restart (e.g. after editing sshd_config)

Anything that changes the system needs root. The script re-runs itself with
sudo when it has to. It handles both the classic ssh.service and the
socket-activated ssh.socket that Ubuntu has used by default since 22.10.

Standard library only; Python 3.7+."""

from __future__ import annotations

import argparse
import getpass
import os
import re
import shlex
import shutil
import subprocess
import sys
from typing import List, NamedTuple, Optional

PACKAGE = "openssh-server"
SERVICE_CANDIDATES = ("ssh.service", "sshd.service")  # Debian/Ubuntu name first
PRIVSEP_DIR = "/run/sshd"  # sshd -t / -T refuse to run without it


# ─── Output ──────────────────────────────────────────────────────────────────

def _use_color(stream) -> bool:
    return (
        getattr(stream, "isatty", lambda: False)()
        and "NO_COLOR" not in os.environ
        and os.environ.get("TERM") != "dumb"
    )


def _unicode_ok(stream) -> bool:
    return "utf" in (getattr(stream, "encoding", "") or "").lower()


def _sym(fancy: str, plain: str) -> str:
    return fancy if _unicode_ok(sys.stdout) else plain


def paint(text: str, code: str, stream=sys.stdout) -> str:
    """Wrap text in an ANSI colour code when the stream is a colour terminal."""
    return f"\033[{code}m{text}\033[0m" if _use_color(stream) else text


def _emit(stream, symbol: str, code: str, msg: str) -> None:
    print(f"{paint(symbol, code, stream)} {msg}", file=stream, flush=True)


def info(msg: str) -> None:
    _emit(sys.stdout, _sym("•", "-"), "36", msg)


def ok(msg: str) -> None:
    _emit(sys.stdout, _sym("✔", "OK"), "32", msg)


def warn(msg: str) -> None:
    _emit(sys.stderr, _sym("!", "WARNING:"), "33;1", msg)


def error(msg: str) -> None:
    _emit(sys.stderr, _sym("✘", "ERROR:"), "31;1", msg)


# ─── Running commands ────────────────────────────────────────────────────────

class SSHCtlError(Exception):
    """A failure we can explain to the user: reported cleanly, exit status 1."""


def run(cmd: List[str], *, check: bool = True, capture: bool = True,
        env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """Run a command. With check=True, a non-zero exit raises SSHCtlError."""
    try:
        result = subprocess.run(cmd, text=True, capture_output=capture, env=env)
    except FileNotFoundError:
        raise SSHCtlError(f"Command not found: {cmd[0]}") from None
    if check and result.returncode != 0:
        msg = f"Command failed (exit {result.returncode}): {shlex.join(cmd)}"
        detail = (result.stderr or result.stdout or "").strip() if capture else ""
        if detail:
            msg += "\n    " + detail.replace("\n", "\n    ")
        raise SSHCtlError(msg)
    return result


def query(cmd: List[str]) -> str:
    """Run a read-only command and return its stripped stdout ('' on any failure)."""
    try:
        return run(cmd, check=False).stdout.strip()
    except SSHCtlError:
        return ""


# ─── System inspection ───────────────────────────────────────────────────────

class Units(NamedTuple):
    service: Optional[str]  # e.g. ssh.service; None if not installed
    socket: Optional[str]   # e.g. ssh.socket; None on non-socket-activated systems

    @property
    def all(self) -> List[str]:
        """Socket first, so it is stopped before it can re-launch the service."""
        return [u for u in (self.socket, self.service) if u]


def unit_exists(unit: str) -> bool:
    listing = query(["systemctl", "list-unit-files", "--no-legend", "--no-pager", unit])
    return any(line.split()[0] == unit for line in listing.splitlines() if line.strip())


def find_units() -> Units:
    for service in SERVICE_CANDIDATES:
        if unit_exists(service):
            socket = service.replace(".service", ".socket")
            return Units(service, socket if unit_exists(socket) else None)
    return Units(None, None)


def active_state(unit: str) -> str:
    return query(["systemctl", "is-active", unit]) or "unknown"


def enabled_state(unit: str) -> str:
    return query(["systemctl", "is-enabled", unit]) or "unknown"


def package_installed() -> bool:
    if not shutil.which("dpkg-query"):
        return False
    return "ok installed" in query(["dpkg-query", "-W", "-f=${Status}", PACKAGE])


def sshd_binary() -> Optional[str]:
    found = shutil.which("sshd")
    if found:
        return found
    for path in ("/usr/sbin/sshd", "/usr/bin/sshd"):
        if os.access(path, os.X_OK):
            return path
    return None


def _ensure_privsep_dir() -> None:
    # On socket-activated Ubuntu this directory only exists once ssh.service has
    # run, and without it "sshd -t" fails with a misleading error.
    try:
        os.makedirs(PRIVSEP_DIR, mode=0o755, exist_ok=True)
    except OSError:
        pass


def sshd_ports() -> List[str]:
    """Ports sshd is configured for (needs root; [] if unknown)."""
    sshd = sshd_binary()
    if not sshd or os.geteuid() != 0:
        return []
    _ensure_privsep_dir()
    ports = {line.split()[1] for line in query([sshd, "-T"]).splitlines()
             if line.startswith("port ")}
    return sorted(ports, key=int)


def local_addresses() -> List[str]:
    addrs = query(["hostname", "-I"]).split()
    ipv4 = [a for a in addrs if ":" not in a]
    return (ipv4 or addrs)[:4]


def in_ssh_session() -> bool:
    return bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"))


# ─── Preconditions ───────────────────────────────────────────────────────────

def check_platform() -> None:
    if not sys.platform.startswith("linux"):
        raise SSHCtlError("This script is for Linux (Ubuntu/Debian).")
    if not shutil.which("systemctl"):
        raise SSHCtlError("systemctl not found. This script needs a systemd-based system.")
    if not os.path.isdir("/run/systemd/system"):
        raise SSHCtlError("systemd is not running here (common in containers and WSL "
                          "without systemd), so services can't be managed.")


def ensure_root(argv: List[str]) -> None:
    """Re-run this script under sudo unless we're already root."""
    if os.geteuid() == 0:
        return
    sudo = shutil.which("sudo")
    if not sudo:
        raise SSHCtlError("Root privileges are needed. Run this script as root.")
    info("Root privileges needed, so re-running with sudo…")
    sys.stdout.flush()
    os.execvp(sudo, [sudo, sys.executable, os.path.abspath(__file__), *argv])


def confirm(question: str) -> bool:
    if not sys.stdin.isatty():
        return False
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


# ─── Actions ─────────────────────────────────────────────────────────────────

def install_package() -> None:
    if not shutil.which("apt-get"):
        raise SSHCtlError(f"{PACKAGE} isn't installed and apt-get isn't available. "
                          f"Install {PACKAGE} with your package manager, then rerun.")
    info(f"Installing {PACKAGE} (this can take a minute)…")
    env = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
    run(["apt-get", "update"], capture=False, env=env)
    run(["apt-get", "install", "-y", PACKAGE], capture=False, env=env)
    ok(f"{PACKAGE} installed")


def validate_config() -> None:
    """Refuse to (re)start sshd with a broken config: that's how people get locked out."""
    sshd = sshd_binary()
    if not sshd:
        return
    _ensure_privsep_dir()
    result = run([sshd, "-t"], check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().replace("\n", "\n    ")
        raise SSHCtlError("sshd configuration is invalid, so the service was left "
                          f"untouched.\n    {detail}")
    ok("sshd configuration is valid")


def start(units: Units, *, at_boot: bool) -> Units:
    if units.service is None:
        if package_installed():
            raise SSHCtlError(f"{PACKAGE} is installed but systemd has no ssh unit for it. "
                              f"Try: sudo apt-get install --reinstall {PACKAGE}")
        install_package()
        units = find_units()
        if units.service is None:
            raise SSHCtlError("The SSH service is still missing after installation.")

    validate_config()

    if at_boot:
        run(["systemctl", "enable", *units.all])
        ok(f"Enabled at boot: {', '.join(units.all)}")

    if active_state(units.service) == "active":
        ok(f"{units.service} is already running")
    else:
        run(["systemctl", "start", units.service])
        ok(f"Started {units.service}")
    return units


def stop(units: Units, *, at_boot: bool) -> Units:
    if units.service is None:
        warn(f"{PACKAGE} isn't installed, so there's nothing to stop.")
        return units

    running = [u for u in units.all if active_state(u) == "active"]
    # Stop the socket too: otherwise the next incoming connection just
    # starts ssh.service again.
    run(["systemctl", "stop", *units.all])
    ok(f"Stopped {', '.join(running)}" if running else "SSH was already stopped")

    if at_boot:
        run(["systemctl", "disable", *units.all])
        ok(f"Disabled at boot: {', '.join(units.all)}")
    return units


def restart(units: Units) -> Units:
    if units.service is None:
        raise SSHCtlError(f"{PACKAGE} isn't installed. Use --start to install and start it.")
    validate_config()
    run(["systemctl", "restart", units.service])
    ok(f"Restarted {units.service}")
    return units


def firewall_hint() -> None:
    """Warn if ufw is active and nothing in it lets SSH through."""
    if os.geteuid() != 0 or not shutil.which("ufw"):
        return
    status = query(["ufw", "status"])
    if not status.startswith("Status: active"):
        return
    ports = sshd_ports() or ["22"]
    allowed = "OpenSSH" in status or any(
        re.search(rf"^{p}(/tcp)?\s", status, re.MULTILINE) for p in ports)
    if not allowed:
        warn("Firewall (ufw) is active and has no rule for SSH, so remote connections "
             "will be blocked.\n  Allow them with:  sudo ufw allow OpenSSH")


# ─── Status report ───────────────────────────────────────────────────────────

_STATE_COLORS = {
    "active": "32", "enabled": "32", "installed": "32",
    "inactive": "2", "disabled": "2",
    "failed": "31;1", "not installed": "33", "masked": "31",
}


def show_status(units: Units, *, verbose: bool = False) -> None:
    rows = [("package", PACKAGE, "installed" if package_installed() else "not installed")]
    for unit in (units.service, units.socket):
        if unit:
            rows.append((unit, active_state(unit), enabled_state(unit)))

    accepting = any(active_state(u) == "active" for u in units.all)
    if units.service is None:
        headline = paint("not installed", "33;1")
    elif accepting:
        headline = paint("accepting connections", "32;1")
    else:
        headline = paint("not accepting connections", "31;1")

    title = f"OpenSSH server {_sym('—', '-')} "
    print()
    print(f"  {paint(title, '1')}{headline}")
    print("  " + _sym("─", "-") * 46)

    w1 = max(len(r[0]) for r in rows) + 3
    w2 = max(len(r[1]) for r in rows) + 3
    for label, col2, col3 in rows:
        c2 = paint(col2.ljust(w2), _STATE_COLORS.get(col2, "0"))
        c3 = paint(col3, _STATE_COLORS.get(col3, "0"))
        print(f"  {label.ljust(w1)}{c2}{c3}")

    if accepting:
        addrs = local_addresses()
        if addrs:
            user = os.environ.get("SUDO_USER") or getpass.getuser()
            ports = sshd_ports()
            port_flag = f" -p {ports[0]}" if ports and ports[0] != "22" else ""
            print(f"  {'connect'.ljust(w1)}{paint(f'ssh {user}@{addrs[0]}{port_flag}', '1')}")
            if len(addrs) > 1:
                print(f"  {'also on'.ljust(w1)}{', '.join(addrs[1:])}")
    print()

    if verbose and units.all:
        subprocess.run(["systemctl", "--no-pager", "status", *units.all])


# ─── Command line ────────────────────────────────────────────────────────────

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]) or "sshctl.py",
        description="Start, stop, restart or inspect the OpenSSH server (Ubuntu/Debian).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s                  show status\n"
            "  %(prog)s --start          install if needed, start, enable at boot\n"
            "  %(prog)s --start --no-boot  start now only\n"
            "  %(prog)s --stop           stop and disable at boot\n"
            "  %(prog)s --restart        apply sshd_config changes safely\n"
        ),
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--start", dest="action", action="store_const", const="start",
                         help="install if necessary, start SSH, and enable it at boot")
    actions.add_argument("--stop", dest="action", action="store_const", const="stop",
                         help="stop SSH and disable it at boot (does not uninstall)")
    actions.add_argument("--restart", dest="action", action="store_const", const="restart",
                         help="check sshd_config, then restart SSH")
    actions.add_argument("--status", dest="action", action="store_const", const="status",
                         help="show status without changing anything (default)")
    parser.add_argument("--no-boot", action="store_true",
                        help="with --start/--stop: leave the start-at-boot setting as it is")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="don't ask for confirmation when stopping SSH over SSH")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="also print the full 'systemctl status' output")
    parser.set_defaults(action="status")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    check_platform()

    if args.action == "stop" and in_ssh_session() and not args.yes:
        warn("You are connected over SSH. Your current session normally survives, "
             "but you won't be able to reconnect until SSH is started again locally.")
        if not confirm("Stop SSH anyway?"):
            info("Cancelled. Nothing was changed. (Use --yes to skip this question.)")
            return 1
        argv.append("--yes")  # don't ask again after the sudo re-run

    if args.action != "status":
        ensure_root(argv)

    units = find_units()
    at_boot = not args.no_boot

    if args.action == "start":
        units = start(units, at_boot=at_boot)
        firewall_hint()
    elif args.action == "stop":
        units = stop(units, at_boot=at_boot)
    elif args.action == "restart":
        units = restart(units)
        firewall_hint()

    show_status(units, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SSHCtlError as exc:
        error(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        error("Interrupted.")
        sys.exit(130)
