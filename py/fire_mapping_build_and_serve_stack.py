#!/usr/bin/env python3
"""20260622 fire_mapping_build_and_serve_stack.py

Call this in ~/refresh_mrap.sh by:
    fire_mapping_build_and_serve_stack.py >> ".log_$(date +%Y%m%d_%H%M%S).txt" 2>&1

Run this AFTER refresh_mrap.sh has finished and a new <yyyymmdd>_mrap.bin
has landed in /data/mrap_bc/.

Steps:
    1. Stop the running fire-mapping web server (it holds /ram/*_stack.bin
       open; the stack must not be rebuilt while it's serving).
    2. Find the most recently DATED *_mrap.bin in /data/mrap_bc (by the
       yyyymmdd filename prefix, not mtime -- regenerated files can have
       out-of-order mtimes).
    3. Run sentinel2_anomaly3 on [pre=composite/median.bin] [post=that
       mrap.bin] -- it always writes ./ratio.bin in the CWD, so we run it
       in a scratch dir and immediately consume the result.
    4. Stack [median.bin] [post mrap.bin] [ratio.bin] -> /ram/<date>_stack.bin
       via raster_stack.py.
    5. Delete the previous day's /ram/<date>_stack.bin (and its .hdr), so
       /ram doesn't fill up -- this is almost certainly tmpfs.
    6. Rewrite the RASTERS=(...) line in run_fire_viirs_web.sh to point at
       the new dated stack file.
    7. Restart the server.

Exits non-zero on any failure. Steps before the server restart try not to
leave things worse than they started -- e.g. a failed rewrite of the
launch script restores its backup.
"""

import glob
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
# Matches any active (non-comment) path line inside RASTERS=( ... ),
# e.g. "    /ram/20260618_stack.bin" or "    /data/mrap_bc/20260618_mrap.bin".
# Deliberately NOT restricted to the /ram/<date>_stack.bin shape, so this
# still works if someone has manually pointed the array at a raw
# .../mrap.bin for testing -- it just replaces whatever single active
# path line it finds.
RASTER_LINE_RE = re.compile(r"^(\s*)(/\S+)\s*$")


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def build_env() -> dict:
    """Current environment with EXTRA_PATH prepended to PATH, matching
    the bash script's PATH export."""
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(EXTRA_PATH + [env.get("PATH", "")])
    return env


def require_on_path(tool: str, env: dict) -> None:
    if shutil.which(tool, path=env["PATH"]) is None:
        die(f"{tool} not found on PATH")


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
    """Best-effort LAN IP, via the same trick the app itself uses at
    startup (open a UDP socket toward an external address; nothing is
    actually sent, this just makes the OS pick the outbound route/IP)."""
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
    """Report the IP/hostname/port the server is (or would be)
    reachable at, regardless of whether it's currently up."""
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
            # Port was free but a matching process existed anyway --
            # e.g. it was stuck shutting down, or about to bind.
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


def run_anomaly(pre_bin: Path, post_bin: Path, scratch_dir: Path) -> Path:
    log("Running sentinel2_anomaly3 ...")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Without --divide, sentinel2_anomaly3 writes ratio.bin/.hdr. Clear
    # out both possible names (plain and --divide's ratio_divide.*)
    # from a previous run before invoking it, in case the mode ever
    # changes again.
    for stem in ("ratio", "ratio_divide"):
        (scratch_dir / f"{stem}.bin").unlink(missing_ok=True)
        (scratch_dir / f"{stem}.hdr").unlink(missing_ok=True)

    subprocess.run(
        ["sentinel2_anomaly3", str(pre_bin), str(post_bin)],
        cwd=scratch_dir, check=True, env=build_env(),
    )

    ratio_bin = scratch_dir / "ratio.bin"
    if not ratio_bin.exists():
        die(f"sentinel2_anomaly3 did not produce {ratio_bin}")
    return ratio_bin


def run_stack(pre_bin: Path, post_bin: Path, ratio_bin: Path,
              stack_out: Path) -> None:
    log(f"Stacking pre + post + anomaly -> {stack_out} ...")
    subprocess.run(
        ["raster_stack.py", str(pre_bin), str(post_bin), str(ratio_bin),
         str(stack_out)],
        check=True, env=build_env(),
    )
    if not stack_out.exists():
        die(f"raster_stack.py did not produce {stack_out}")
    log(f"Stack written: {stack_out}")


def delete_previous_stacks(ram_dir: Path, keep: Path) -> None:
    """Delete every <yyyymmdd>_stack.bin/.hdr in ram_dir except `keep`,
    so /ram (almost certainly tmpfs) doesn't fill up with old stacks."""
    removed_any = False
    for f in ram_dir.glob("*_stack.bin"):
        if f == keep:
            continue
        for path in (f, f.with_suffix(".hdr")):
            if path.exists():
                log(f"Deleting old stack file: {path}")
                path.unlink()
                removed_any = True
    if not removed_any:
        log("No previous stack files to delete.")


def delete_scratch(scratch_dir: Path) -> None:
    """Delete sentinel2_anomaly3's ratio.bin/.hdr (and the scratch dir
    itself) now that the stack has consumed them -- they're only an
    intermediate, and /ram is almost certainly tmpfs."""
    if scratch_dir.exists():
        log(f"Deleting anomaly scratch dir: {scratch_dir}")
        shutil.rmtree(scratch_dir)


def update_rasters_line(script_path: Path, stack_out: Path) -> None:
    """Rewrite the single active (non-comment) path inside
    RASTERS=( ... ) in script_path to point at stack_out.

    Scoped strictly to lines between "RASTERS=(" and the closing ")",
    so this can't accidentally touch OUT_ROOT, LAADS_TOKEN_FILE, or
    anything else in the script. Works regardless of what the current
    active line looks like -- a /ram/<date>_stack.bin from a previous
    run, or a raw /data/mrap_bc/<date>_mrap.bin someone pointed it at
    by hand for testing -- as long as there's exactly one uncommented
    path line in the array.
    """
    log(f"Updating RASTERS in {script_path} to point at {stack_out} ...")
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
            trailing = line[m.end(2):]  # preserve trailing newline etc.
            new_lines.append(f"{indent}{stack_out}{trailing}")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        die(f"could not find an active (uncommented) path line inside "
            f"RASTERS=( ... ) in {script_path} -- leaving it unchanged. "
            f"Check that the array has exactly one uncommented path.")

    script_path.write_text("".join(new_lines))

    if str(stack_out) not in script_path.read_text():
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
            start_new_session=True,  # detach, survives this script exiting
        )

    time.sleep(2)
    if port_in_use(SERVER_PORT):
        log(f"Server is back up.")
        report_server_address(SERVER_PORT)
    else:
        print(f"WARNING: server may not have started -- check {log_name}",
              file=sys.stderr)


def main() -> None:
    if not PRE_BIN.exists():
        die(f"pre-image not found: {PRE_BIN}")

    env = build_env()
    require_on_path("sentinel2_anomaly3", env)
    require_on_path("raster_stack.py", env)

    date_str, post_bin = find_latest_mrap(MRAP_DIR)
    stack_out = RAM_DIR / f"{date_str}_stack.bin"

    log(f"Pre-image  : {PRE_BIN}")
    log(f"Post-image : {post_bin} (date={date_str})")
    log(f"Stack out  : {stack_out}")

    stop_server()

    ratio_bin = run_anomaly(PRE_BIN, post_bin, SCRATCH_DIR)
    run_stack(PRE_BIN, post_bin, ratio_bin, stack_out)
    delete_scratch(SCRATCH_DIR)
    delete_previous_stacks(RAM_DIR, keep=stack_out)

    update_rasters_line(SERVER_DIR / SERVER_SCRIPT, stack_out)
    start_server(SERVER_DIR, SERVER_SCRIPT)

    log("Done.")


if __name__ == "__main__":
    main()

