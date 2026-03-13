#!/usr/bin/env python3
"""
Build a multi-band ENVI product from two Sentinel-2 scenes (pre/post fire).

Usage
-----
    python write_envi_product.py  pre_image.bin  post_image.bin
    python write_envi_product.py  pre.bin  post.bin  --output /some/dir/
    python write_envi_product.py  pre.bin  post.bin  -o /some/dir/custom_name.bin

Output filename (auto-generated when --output is omitted or points to a dir):
    {LEVEL}_{pre_datetime_UTC}_{post_datetime_UTC}.bin
    e.g.  L1C_20250609T192931_20250904T191931.bin

Extending the product
---------------------
Write a new function with the signature:

    def my_product(pre_obj, post_obj, pre_dat, post_dat):
        ...
        return array, ["bandname1", "bandname2", ...]

Then append it to BAND_FUNCTIONS at the bottom of this file.
The array may be 2-D (H, W) or 3-D (H, W, K); names must have K entries.
"""

import sys
import re
import argparse
from pathlib import Path

import numpy as np

# ── project root (parent of notebooks/) ───────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from misc.sen2 import writeENVI, read_timestamp_filename
from fire_mapping.raster import Raster
from fire_mapping.change_detection import change_detection
from fire_mapping.barc import dNBR


# ── filename helpers ───────────────────────────────────────────────────────────

def _parse_level(filename: str) -> str:
    """Extract processing level (e.g. L1C, L2A) from a Sentinel-2 filename."""
    m = re.search(r"MSI(L\w+?)_", Path(filename).name)
    return m.group(1) if m else "L??"


def _auto_output_path(pre_file: str, post_file: str, out_dir: Path) -> Path:
    level  = _parse_level(pre_file)
    pre_ts = read_timestamp_filename(pre_file)
    pst_ts = read_timestamp_filename(post_file)
    return out_dir / f"{level}_{pre_ts}_{pst_ts}.bin"


# ── band-generation functions ──────────────────────────────────────────────────
#
#  Signature:
#      func(pre_obj, post_obj, pre_dat, post_dat) -> (np.ndarray, list[str])
#
#  • pre_obj / post_obj  — Raster instances  (.get_band(n) returns a 2-D slice)
#  • pre_dat / post_dat  — 3-D float32 arrays (H, W, K) with all bands
#  • Returned array may be 2-D (H, W) or 3-D (H, W, K)
#  • Returned list must have exactly K names
#
#  Band names must match the HDR description format: prefix + band label,
#  e.g. "preB12", "pstB11", "diffB9", "dnbr11".


def _bands_pre(pre_obj: Raster, post_obj: Raster,
               pre_dat: np.ndarray, post_dat: np.ndarray):
    """All pre-fire bands: preB12, preB11, ..."""
    names = [f"pre{pre_obj.band_info_list[i]}" for i in range(pre_dat.shape[2])]
    return pre_dat, names


def _bands_post(pre_obj: Raster, post_obj: Raster,
                pre_dat: np.ndarray, post_dat: np.ndarray):
    """All post-fire bands: pstB12, pstB11, ..."""
    names = [f"pst{post_obj.band_info_list[i]}" for i in range(post_dat.shape[2])]
    return post_dat, names


def _bands_diff(pre_obj: Raster, post_obj: Raster,
                pre_dat: np.ndarray, post_dat: np.ndarray):
    """Normalised difference (post - pre) / (post + pre): diffB12, diffB11, ..."""
    diff  = change_detection(pre_X=pre_dat, post_X=post_dat)
    names = [f"diff{pre_obj.band_info_list[i]}" for i in range(pre_dat.shape[2])]
    return diff, names


def _band_dnbr11(pre_obj: Raster, post_obj: Raster,
                 pre_dat: np.ndarray, post_dat: np.ndarray):
    """dNBR using B8 (NIR) and B11 (SWIR)."""
    _, _, dnbr = dNBR(
        NIR_pre=pre_obj.get_band(8),   SWIR_pre=pre_obj.get_band(11),
        NIR_post=post_obj.get_band(8), SWIR_post=post_obj.get_band(11),
    )
    return dnbr, ["dnbr11"]


def _band_dnbr12(pre_obj: Raster, post_obj: Raster,
                 pre_dat: np.ndarray, post_dat: np.ndarray):
    """dNBR using B8 (NIR) and B12 (SWIR)."""
    _, _, dnbr = dNBR(
        NIR_pre=pre_obj.get_band(8),   SWIR_pre=pre_obj.get_band(12),
        NIR_post=post_obj.get_band(8), SWIR_post=post_obj.get_band(12),
    )
    return dnbr, ["dnbr12"]


# ── registry ───────────────────────────────────────────────────────────────────
# Append new functions here to include them in the output file.

BAND_FUNCTIONS = [
    _bands_pre,
    _bands_post,
    _bands_diff
]


# ── stack builder ──────────────────────────────────────────────────────────────

def build_stack(
        pre_obj: Raster,
        post_obj: Raster,
        pre_dat: np.ndarray,
        post_dat: np.ndarray,
):
    """Call every registered band function and concatenate results depth-wise."""
    arrays = []
    names  = []

    for fn in BAND_FUNCTIONS:
        arr, band_names = fn(pre_obj, post_obj, pre_dat, post_dat)

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]

        if arr.shape[2] != len(band_names):
            raise ValueError(
                f"{fn.__name__}: returned {arr.shape[2]} band(s) "
                f"but {len(band_names)} name(s)."
            )

        arrays.append(arr)
        names.extend(band_names)

    return np.concatenate(arrays, axis=2), names


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build a multi-band ENVI product from two Sentinel-2 scenes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pre_image",  help="Pre-fire ENVI .bin file")
    parser.add_argument("post_image", help="Post-fire ENVI .bin file")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=(
            "Output path. Pass a directory to auto-name the file inside it, "
            "or a full .bin path to use that name exactly. "
            "Defaults to the current working directory."
        ),
    )
    args = parser.parse_args()

    pre_file  = args.pre_image
    post_file = args.post_image

    # resolve output path
    if args.output is None:
        out_path = _auto_output_path(pre_file, post_file, Path.cwd())
    else:
        out = Path(args.output)
        out_path = _auto_output_path(pre_file, post_file, out) if out.is_dir() else out

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # load rasters
    print(f"Pre-fire  : {pre_file}")
    pre_obj  = Raster(pre_file)
    pre_dat  = pre_obj.read_bands("all")

    print(f"Post-fire : {post_file}")
    post_obj = Raster(post_file)
    post_dat = post_obj.read_bands("all")

    # build stack
    print("Building band stack...")
    stack, band_names = build_stack(pre_obj, post_obj, pre_dat, post_dat)

    print(f"  Shape : {stack.shape}")
    for i, name in enumerate(band_names, 1):
        print(f"  [{i:>2}]  {name}")

    # write
    print(f"\nWriting -> {out_path}")
    writeENVI(
        output_filename=str(out_path),
        data=stack,
        mode="new",
        ref_filename=pre_file,
        band_names=band_names,
    )
    print("Done.")


if __name__ == "__main__":
    main()
