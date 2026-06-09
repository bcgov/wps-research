# -*- coding: utf-8 -*-
"""20260692 cuda based ccdc implementation
ccdc.py
=======
Companion launcher for ccdc.cu (miniCCDC CUDA change detection).

Lives  : ~/GitHub/wps-research/py/ccdc.py
Expects: ~/GitHub/wps-research/cpp/ccdc.cu   (source)
         ccdc                                  (compiled binary, on PATH)

Steps
-----
1. Compile ~/GitHub/wps-research/cpp/ccdc.cu  -> ccdc  (skipped if
   the binary is newer than the source, unless --force-compile is given).
2. Glob S2*.bin files in the current working directory, sort by
   acquisition date (field 3, left of 'T'), write a temporary files.txt.
3. Read the ENVI header of the first file to extract grid dimensions.
4. Call the ccdc executable with default (or user-supplied) parameters.

Usage
-----
    # from inside the directory that holds your S2*.bin files:
    python3 ~/GitHub/wps-research/py/ccdc.py

    # override parameters:
    python3 ~/GitHub/wps-research/py/ccdc.py \
        --output_dir ./ccdc_out  \
        --min_history 24         \
        --consecutive 5          \
        --alpha 0.99             \
        --harmonics 2            \
        --refit 8                \
        --min_years 1.5          \
        --min_hist_years 0.8     \
        --threads 256            \
        --force-compile

    # point at a different data directory:
    python3 ~/GitHub/wps-research/py/ccdc.py --data_dir /mnt/ramdisk/s2
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path.home() / "GitHub" / "wps-research"
CPP_DIR    = REPO_ROOT / "cpp"
SOURCE_CU  = CPP_DIR / "ccdc.cu"
BINARY     = CPP_DIR / "ccdc"          # also assumed to be on PATH

# ---------------------------------------------------------------------------
# Filename regex  (matches S2A/B/C MSIL2A standard naming)
# ---------------------------------------------------------------------------
S2_RE = re.compile(r"S2[ABC]_MSIL2A_(\d{8})T\d{6}_")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_s2_files(data_dir: Path) -> list[tuple[str, Path]]:
    """
    Return list of (YYYYMMDD, Path) sorted by acquisition date.
    Only files matching S2*.bin with a parseable date are included.
    """
    pairs = []
    for f in data_dir.glob("S2*.bin"):
        m = S2_RE.search(f.name)
        if m:
            pairs.append((m.group(1), f.resolve()))
        else:
            print(f"  [SKIP] cannot parse date from {f.name}")
    pairs.sort(key=lambda x: x[0])
    return pairs


def read_envi_dims(bin_path: Path) -> tuple[int, int, int]:
    """
    Return (lines, samples, bands) by parsing the companion .hdr file.
    Raises FileNotFoundError / ValueError if the header is missing or malformed.
    """
    hdr = bin_path.with_suffix(".hdr")
    if not hdr.exists():
        hdr = bin_path.with_name(bin_path.name + ".hdr")
    if not hdr.exists():
        raise FileNotFoundError(f"No .hdr found for {bin_path}")

    dims = {}
    with open(hdr, "r") as fh:
        for line in fh:
            for key in ("lines", "samples", "bands"):
                if line.strip().lower().startswith(key):
                    parts = line.split("=")
                    if len(parts) == 2:
                        try:
                            dims[key] = int(parts[1].strip())
                        except ValueError:
                            pass

    for key in ("lines", "samples", "bands"):
        if key not in dims:
            raise ValueError(f"Could not parse '{key}' from {hdr}")

    return dims["lines"], dims["samples"], dims["bands"]


def needs_compile(source: Path, binary: Path) -> bool:
    """Return True if the binary is missing or older than the source."""
    if not binary.exists():
        return True
    return source.stat().st_mtime > binary.stat().st_mtime


def compile_cuda(source: Path, binary: Path) -> None:
    """
    Compile source with nvcc targeting sm_89 (L40s).
    Exits the script on failure.
    """
    cmd = [
        "nvcc",
        "-O3",
        "-arch=sm_89",
        "-o", str(binary),
        str(source),
        "-lgdal",
        "-I/usr/include/gdal",
    ]
    print("Compiling:")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(binary.parent))
    if result.returncode != 0:
        print(f"\nERROR: nvcc returned {result.returncode}. "
              f"Fix compilation errors and retry.")
        sys.exit(result.returncode)
    print(f"  -> {binary}  [OK]\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compile ccdc.cu and run miniCCDC on local S2*.bin files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- paths ---
    ap.add_argument("--data_dir",     default=".",
                    help="Directory containing S2*.bin files "
                         "(defaults to current working directory)")
    ap.add_argument("--output_dir",   default="./ccdc_out",
                    help="Directory for output ENVI rasters")
    ap.add_argument("--files_txt",    default=None,
                    help="Path for the sorted file list (default: temp file)")

    # --- compile ---
    ap.add_argument("--force-compile", action="store_true",
                    help="Recompile even if binary is up to date")
    ap.add_argument("--skip-compile",  action="store_true",
                    help="Never compile; assume binary already exists")

    # --- algorithm (mirrors ccdc.cu defaults) ---
    ap.add_argument("--min_history",    type=int,   default=24,
                    help="Min observations for initial model fit")
    ap.add_argument("--consecutive",    type=float, default=5.0,
                    help="CUSUM threshold multiplier k")
    ap.add_argument("--alpha",          type=float, default=0.99,
                    help="Chi-squared significance level")
    ap.add_argument("--harmonics",      type=int,   default=2,
                    help="Number of seasonal harmonics")
    ap.add_argument("--refit",          type=int,   default=8,
                    help="Refit model every N accepted observations")
    ap.add_argument("--cusum_k",        type=float, default=0.5,
                    help="CUSUM inlier slack (fraction of chi2_crit)")
    ap.add_argument("--min_years",      type=float, default=1.5,
                    help="Minimum years of valid data to process a pixel")
    ap.add_argument("--min_hist_years", type=float, default=0.8,
                    help="Minimum years spanned by initial history window")
    ap.add_argument("--threads",        type=int,   default=256,
                    help="CUDA threads per block")

    args = ap.parse_args()

    data_dir   = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # ------------------------------------------------------------------
    # 1. Compile (if needed)
    # ------------------------------------------------------------------
    if not SOURCE_CU.exists():
        print(f"ERROR: source not found: {SOURCE_CU}")
        sys.exit(1)

    if args.skip_compile:
        print(f"Skipping compilation (--skip-compile).  "
              f"Assuming {BINARY.name} is on PATH.")
    elif args.force_compile or needs_compile(SOURCE_CU, BINARY):
        compile_cuda(SOURCE_CU, BINARY)
    else:
        age_src = SOURCE_CU.stat().st_mtime
        age_bin = BINARY.stat().st_mtime
        print(f"Binary is up to date (skip --force-compile to recompile).")
        print(f"  source : {SOURCE_CU}")
        print(f"  binary : {BINARY}\n")

    # ------------------------------------------------------------------
    # 2. Discover and sort S2 files
    # ------------------------------------------------------------------
    print(f"Scanning for S2*.bin files in {data_dir} ...")
    pairs = find_s2_files(data_dir)

    if not pairs:
        print("ERROR: no S2*.bin files found. "
              "Check --data_dir or run from the data directory.")
        sys.exit(1)

    print(f"  Found {len(pairs)} files  "
          f"[{pairs[0][0]}  ->  {pairs[-1][0]}]")

    # ------------------------------------------------------------------
    # 3. Read grid dimensions from first file header
    # ------------------------------------------------------------------
    first_bin = pairs[0][1]
    try:
        n_lines, n_samples, n_bands = read_envi_dims(first_bin)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR reading header: {exc}")
        sys.exit(1)

    print(f"  Grid: {n_lines} lines x {n_samples} samples x {n_bands} bands")

    # ------------------------------------------------------------------
    # 4. Write files.txt
    # ------------------------------------------------------------------
    # Use a caller-specified path or a named temp file that persists
    # long enough for the subprocess to read it.
    if args.files_txt:
        files_txt_path = Path(args.files_txt)
        files_txt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = None
    else:
        # NamedTemporaryFile with delete=False so the subprocess can open it
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="ccdc_files_"
        )
        files_txt_path = Path(tmp.name)
        tmp_file = tmp

    try:
        with open(files_txt_path, "w") as fh:
            for date_str, path in pairs:
                fh.write(f"{date_str} {path}\n")
        print(f"  File list written -> {files_txt_path}  "
              f"({len(pairs)} entries)\n")

        # ------------------------------------------------------------------
        # 5. Create output directory
        # ------------------------------------------------------------------
        output_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # 6. Build and run the ccdc command
        # ------------------------------------------------------------------
        cmd = [
            "ccdc",                              # assumed on PATH
            "--input_list",    str(files_txt_path),
            "--output_dir",    str(output_dir),
            "--lines",         str(n_lines),
            "--samples",       str(n_samples),
            "--bands",         str(n_bands),
            "--min_history",   str(args.min_history),
            "--consecutive",   str(args.consecutive),
            "--alpha",         str(args.alpha),
            "--harmonics",     str(args.harmonics),
            "--refit",         str(args.refit),
            "--cusum_k",       str(args.cusum_k),
            "--min_years",     str(args.min_years),
            "--min_hist_years",str(args.min_hist_years),
            "--threads",       str(args.threads),
        ]

        print("Running ccdc:")
        print("  " + " ".join(cmd))
        print()

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\nERROR: ccdc exited with code {result.returncode}")
            sys.exit(result.returncode)

        print(f"\nDone.  Outputs in: {output_dir}")

    finally:
        # Clean up temp file if we created one
        if tmp_file is not None:
            try:
                os.unlink(files_txt_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()


