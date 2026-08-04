#!/usr/bin/env python3
"""20260622 fire_mapping_build_and_serve_stack.py

Call this in ~/refresh_mrap.sh by:
    fire_mapping_build_and_serve_stack.py >> ".log_$(date +%Y%m%d_%H%M%S).txt" 2>&1

Run this AFTER refresh_mrap.sh has finished and a new <yyyymmdd>_mrap.bin
has landed in /data/mrap_bc/.

20260804: STOPPED PRE-BUILDING THE PROVINCE-WIDE STACK.
----------------------------------------------------------------------
This script used to run sentinel2_anomaly3 over the whole province and
stack pre + post + anomaly into /ram/<date>_stack.bin. That file was
~307 GB -- three times the size of the 103 GB mosaic it came from --
and essentially all of it was never read: fires are mapped on small
AOIs, and each one only ever used its own little window.

The stack is now generated per-AOI, on demand, when a fire is created
through the new_fire interface (see aoi_stack.py in the web app). Each
AOI stack is written to /ram/<postdate>_stack_<identifier>.bin, is the
same kind of product this script used to cut out of the province-wide
stack, and costs megabytes instead of hundreds of gigabytes. /ram is
tmpfs, so those per-AOI stacks are expendable; the web app rebuilds any
that go missing after a reboot.

What this script does now:
    1. Find the most recently DATED *_mrap.bin in /data/mrap_bc (by the
       yyyymmdd filename prefix, not mtime -- regenerated files can have
       out-of-order mtimes).
    2. Stop the running fire-mapping web server.
    3. Delete any leftover province-wide /ram/<date>_stack.bin from the
       old workflow, and any stale per-AOI stacks whose date prefix no
       longer matches the current mosaic (their anomaly bands were
       computed against a superseded post image).
    4. Rewrite the RASTERS=(...) line in run_fire_viirs_web.sh to point
       at the new mosaic itself. The web app derives its overview PNGs
       (both pyramid levels) from this file directly.
    5. Restart the server.

Exits non-zero on any failure.
"""

import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

MRAP_DIR = Path("/data/mrap_bc")
COMPOSITE_DIR = MRAP_DIR / "composite"
PRE_BIN = COMPOSITE_DIR / "median.bin"
RAM_DIR = Path("/ram")
SCRATCH_DIR = RAM_DIR / ".anomaly_scratch"
SERVER_DIR = Path("/home/ash/GitHub/wps-research/data/bill")
SERVER_SCRIPT = "run_fire_viirs_web.sh"
SERVER_PORT = 8765

EXTRA_PATH = [
    "/usr/local/bin",
    "/home/ash/GitHub/wps-research/cpp",
    "/home/ash/GitHub/bin/bin",
]

MRAP_NAME_RE = re.compile(r"^(\d{8})_mrap\.bin$")
# Province-wide stacks from the retired workflow: <date>_stack.bin
OLD_STACK_RE = re.compile(r"^(\d{8})_stack\.bin$")
# Per-AOI stacks from the new workflow: <date>_stack_<identifier>.bin
AOI_STACK_RE = re.compile(r"^(\d{8})_stack_(.+)\.bin$")

RASTER_LINE_RE = re.compile(r"^(\s*)(/\S+)\s*$")


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def build_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(EXTRA_PATH + [env.get("PATH", "")])
    return env


def find_latest_mrap(mrap_dir: Path) -> tuple[str, Path]:
    """Return (yyyymmdd, path) for the *_mrap.bin with the largest date
    prefix. Uses the filename, not mtime, since regenerated files can
    land out of date order."""
    candidates = []
    for f in mrap_dir.glob("*_mrap.bin"):
        m = MRAP_NAME_RE.match(f.name)
        if m:
            candidates.append((m.group(1), f))

    if not candidates:
        die(f"no <yyyymmdd>_mrap.bin files found in {mrap_dir}")

    date_str, path = max(candidates, key=lambda pair: pair[0])

    hdr = path.with_suffix(".hdr")
    if not hdr.exists():
        die(f"missing header file {hdr} for {path}")

    return date_str, path


def port_in_use(port: int) -> bool:
    result = subprocess.run(
        ["ss", "-ltn", f"( sport = :{port} )"],
        capture_output=True, text=True,
    )
    return f":{port}" in result.stdout


def lan_ip() -> str | None:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def report_server_address(port: int) -> None:
    import socket
    hostname = socket.gethostname()
    ip = lan_ip()
    if ip:
        log(f"  server address: http://{ip}:{port}  (http://{hostname}:{port})")
    else:
        log(f"  server address: http://{hostname}:{port}  "
            f"(could not determine LAN IP)")


def stop_server() -> None:
    log("Checking fire-mapping server status ...")
    report_server_address(SERVER_PORT)
    was_up = port_in_use(SERVER_PORT)

    if was_up:
        log(f"  server IS UP (port {SERVER_PORT} is in use)")
    else:
        log(f"  server is NOT running (port {SERVER_PORT} is free)")

    result = subprocess.run(["pkill", "-f", "batch_fire_mapping_viirs_web"])
    pkill_matched = (result.returncode == 0)

    if not was_up:
        if pkill_matched:
            log("  no listener on the port, but pkill matched a process "
                "and signalled it anyway")
        else:
            log("  nothing to stop.")
        return

    if not pkill_matched:
        die("server appears to be up (port in use) but no "
            "batch_fire_mapping_viirs_web process was found to kill -- "
            "something else may be holding the port")

    log("  sent SIGTERM to batch_fire_mapping_viirs_web process(es); "
        "waiting for it to exit ...")

    for _ in range(30):
        if not port_in_use(SERVER_PORT):
            log(f"  confirmed: server killed successfully, "
                f"port {SERVER_PORT} is now free")
            return
        time.sleep(1)

    die(f"server did not shut down: port {SERVER_PORT} still in use "
        f"30s after SIGTERM")


def cleanup_ram(ram_dir: Path, current_date: str) -> None:
    """Free the ramdisk of everything the new workflow does not need.

    Three categories:
      * province-wide <date>_stack.bin from the retired workflow -- the
        ~307 GB files this change exists to eliminate;
      * per-AOI stacks whose date prefix is not the current mosaic's,
        since their anomaly bands were computed against a superseded
        post image and would be silently wrong if reused;
      * the old sentinel2_anomaly3 scratch dir.

    Per-AOI stacks matching the *current* date are kept: they are still
    valid, and keeping them means fires mapped earlier today do not have
    to rebuild.
    """
    freed = 0

    for f in sorted(ram_dir.glob("*_stack.bin")):
        if not OLD_STACK_RE.match(f.name):
            continue
        for path in (f, f.with_suffix(".hdr"),
                     Path(str(f) + ".aux.xml"),
                     Path(str(f.with_suffix(".hdr")) + ".bak")):
            if path.exists():
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                log(f"Deleting province-wide stack from the old "
                    f"workflow: {path}")
                try:
                    path.unlink()
                    freed += size
                except OSError as exc:
                    log(f"  WARNING: could not delete {path}: {exc}")

    for f in sorted(ram_dir.glob("*_stack_*.bin")):
        m = AOI_STACK_RE.match(f.name)
        if not m:
            continue
        if m.group(1) == current_date:
            log(f"Keeping current-date AOI stack: {f.name}")
            continue
        for path in (f, f.with_suffix(".hdr"),
                     Path(str(f) + ".aux.xml")):
            if path.exists():
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                log(f"Deleting stale AOI stack (built against "
                    f"{m.group(1)}, current is {current_date}): {path}")
                try:
                    path.unlink()
                    freed += size
                except OSError as exc:
                    log(f"  WARNING: could not delete {path}: {exc}")

    if SCRATCH_DIR.exists():
        log(f"Deleting anomaly scratch dir: {SCRATCH_DIR}")
        shutil.rmtree(SCRATCH_DIR, ignore_errors=True)

    if freed:
        log(f"Freed {freed / (1024 ** 3):.1f} GiB on {ram_dir}.")
    else:
        log("Nothing to clean up on the ramdisk.")


def update_rasters_line(script_path: Path, raster_path: Path) -> None:
    """Rewrite the single active (non-comment) path inside
    RASTERS=( ... ) in script_path to point at raster_path.

    Scoped strictly to lines between "RASTERS=(" and the closing ")",
    so this can't accidentally touch OUT_ROOT, LAADS_TOKEN_FILE, or
    anything else in the script.
    """
    log(f"Updating RASTERS in {script_path} to point at {raster_path} ...")
    backup_path = script_path.with_suffix(script_path.suffix + ".bak")
    shutil.copyfile(script_path, backup_path)

    lines = script_path.read_text().splitlines(keepends=True)
    new_lines = []
    in_rasters_block = False
    replaced = False

    for line in lines:
        stripped = line.strip()

        if not in_rasters_block:
            new_lines.append(line)
            if stripped.startswith("RASTERS="):
                in_rasters_block = True
            continue

        if stripped == ")":
            in_rasters_block = False
            new_lines.append(line)
            continue

        m = RASTER_LINE_RE.match(line)
        if m and not stripped.startswith("#"):
            if replaced:
                die(f"found more than one active path line inside "
                    f"RASTERS=( ... ) in {script_path} -- expected "
                    f"exactly one, refusing to guess which to replace")
            indent = m.group(1)
            trailing = line[m.end(2):]
            new_lines.append(f"{indent}{raster_path}{trailing}")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        die(f"could not find an active (uncommented) path line inside "
            f"RASTERS=( ... ) in {script_path} -- leaving it unchanged. "
            f"Check that the array has exactly one uncommented path.")

    script_path.write_text("".join(new_lines))

    if str(raster_path) not in script_path.read_text():
        log(f"Restoring backup of {script_path} after failed update")
        shutil.copyfile(backup_path, script_path)
        die(f"failed to update RASTERS line in {script_path}")

    log("  updated.")


def start_server(server_dir: Path, server_script: str) -> None:
    log("Starting server ...")
    log_name = server_dir / f".server_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(log_name, "ab") as log_fh:
        subprocess.Popen(
            ["./" + server_script],
            cwd=server_dir,
            stdout=log_fh,
            stderr=log_fh,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    time.sleep(2)
    if port_in_use(SERVER_PORT):
        log(f"Server is back up.")
        report_server_address(SERVER_PORT)
    else:
        print(f"WARNING: server may not have started -- check {log_name}",
              file=sys.stderr)


def main() -> None:
    # The median composite is not consumed here any more, but the web
    # app needs it for every AOI stack it builds. Failing loudly now is
    # far better than every fire creation failing later.
    if not PRE_BIN.exists():
        die(f"pre-image not found: {PRE_BIN} "
            f"(the web app needs this to build per-AOI stacks)")

    date_str, post_bin = find_latest_mrap(MRAP_DIR)

    log(f"Pre-image (median, used per-AOI by the web app): {PRE_BIN}")
    log(f"Post-image (mosaic served to the web app)      : {post_bin}")
    log(f"Date                                            : {date_str}")
    log("No province-wide stack is built; per-AOI stacks are generated "
        "on demand by the web app.")

    stop_server()
    cleanup_ram(RAM_DIR, date_str)

    update_rasters_line(SERVER_DIR / SERVER_SCRIPT, post_bin)
    start_server(SERVER_DIR, SERVER_SCRIPT)

    log("Done.")


if __name__ == "__main__":
    main()
