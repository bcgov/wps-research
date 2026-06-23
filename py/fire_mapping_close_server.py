#!/usr/bin/env python3
"""fire_mapping_close_server.py

Standalone: find and kill the running batch_fire_mapping_viirs_web
server, if any. Same detection/reporting logic as the "stop server"
step inside fire_mapping_build_and_serve_stack.py, pulled out on its
own for cases where you just want to shut the server down without
rebuilding the stack.

Reports the server's address up front (whether it's up or not), reports
whether it was found running, and reports whether the kill actually
succeeded -- in every case, not just the happy path.

Exit codes:
    0  server was not running, or was running and was killed successfully
    1  something went wrong (couldn't confirm shutdown, conflicting
       process/port state, etc.)
"""

import socket
import subprocess
import sys
import time
from datetime import datetime

SERVER_PORT = 8765
PROCESS_PATTERN = "batch_fire_mapping_viirs_web"
SHUTDOWN_TIMEOUT_S = 30


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def port_in_use(port: int) -> bool:
    result = subprocess.run(
        ["ss", "-ltn", f"( sport = :{port} )"],
        capture_output=True, text=True,
    )
    return f":{port}" in result.stdout


def lan_ip() -> str | None:
    """Best-effort LAN IP, via the same trick the app itself uses at
    startup (open a UDP socket toward an external address; nothing is
    actually sent, this just makes the OS pick the outbound route/IP)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def report_server_address(port: int) -> None:
    """Report the IP/hostname/port the server is (or would be)
    reachable at, regardless of whether it's currently up."""
    hostname = socket.gethostname()
    ip = lan_ip()
    if ip:
        log(f"  server address: http://{ip}:{port}  (http://{hostname}:{port})")
    else:
        log(f"  server address: http://{hostname}:{port}  "
            f"(could not determine LAN IP)")


def close_server() -> None:
    log("Checking fire-mapping server status ...")
    report_server_address(SERVER_PORT)
    was_up = port_in_use(SERVER_PORT)

    if was_up:
        log(f"  server IS UP (port {SERVER_PORT} is in use)")
    else:
        log(f"  server is NOT running (port {SERVER_PORT} is free)")

    result = subprocess.run(["pkill", "-f", PROCESS_PATTERN])
    pkill_matched = (result.returncode == 0)

    if not was_up:
        if pkill_matched:
            # Port was free but a matching process existed anyway --
            # e.g. it was stuck shutting down, or about to bind.
            log("  no listener on the port, but pkill matched a process "
                "and signalled it anyway")
        else:
            log("  nothing to stop.")
        return

    if not pkill_matched:
        die("server appears to be up (port in use) but no "
            f"{PROCESS_PATTERN} process was found to kill -- "
            "something else may be holding the port")

    log("  sent SIGTERM to matching process(es); waiting for it to exit ...")

    for _ in range(SHUTDOWN_TIMEOUT_S):
        if not port_in_use(SERVER_PORT):
            log(f"  confirmed: server killed successfully, "
                f"port {SERVER_PORT} is now free")
            return
        time.sleep(1)

    die(f"server did not shut down: port {SERVER_PORT} still in use "
        f"{SHUTDOWN_TIMEOUT_S}s after SIGTERM")


if __name__ == "__main__":
    close_server()

