"""20260502 run CCDC algorithm. Ref: 
    https://github.com/SashaNasonova/changeDetection
run_ccdc_s2.py
==============================================

e.g.: 
python3 run_ccdc_s2.py \
    --input_dir ./ \
    --output_dir ./ccdc/ \
    --n_workers 32

With a whole bunch of data like this in the folder ( lots more ):
S2B_MSIL2A_20240210T193539_N0510_R142_T10VFK_20240210T212916.bin
S2C_MSIL2A_20260519T190911_N0512_R056_T10VFK_20260519T234012.bin
S2B_MSIL2A_20240214T191509_N0510_R056_T10VFK_20240214T210710.bin
S2C_MSIL2A_20260522T191911_N0512_R099_T10VFK_20260522T225518.bin
S2B_MSIL2A_20240217T192459_N0510_R099_T10VFK_20240217T220531.bin
S2C_MSIL2A_20260525T192901_N0512_R142_T10VFK_20260525T223713.bin
S2B_MSIL2A_20240220T193439_N0510_R142_T10VFK_20240220T213038.bin

Fixes:
- correct dtype handling for pyxccd
- correct pos indexing (1-based)
- multiprocessing safety improvements
- robust debug logging
- stricter input validation
"""

import argparse
import re
import sys
import time
from datetime import date, datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import spectral.io.envi as envi
from tqdm import tqdm
from pyxccd import cold_detect_flex


# ---------------------------
# NODATA
# ---------------------------
NODATA_INT16 = -9999
NODATA_INT32 = -9999
NODATA_FLOAT = np.nan


# ---------------------------
# filename regex
# ---------------------------
S2_FILENAME_RE = re.compile(r"S2[ABC]_MSIL2A_(\d{8})T\d{6}_")


def parse_date_from_filename(path: Path) -> date:
    m = S2_FILENAME_RE.search(path.name)
    if not m:
        raise ValueError(path.name)
    return datetime.strptime(m.group(1), "%Y%m%d").date()


def date_to_ordinal(d: date) -> int:
    return d.toordinal()


# ---------------------------
# ENVI loader
# ---------------------------
def read_envi_bsq(bin_path: Path):
    hdr_path = bin_path.with_suffix(".hdr")
    if not hdr_path.exists():
        hdr_path = bin_path.with_name(bin_path.name + ".hdr")

    img = envi.open(str(hdr_path), str(bin_path))
    arr = np.array(img.load(), dtype=np.float32)  # (lines, samples, bands)
    return np.moveaxis(arr, -1, 0)  # (bands, lines, samples)


# ---------------------------
# georef
# ---------------------------
def get_ref_geotransform(bin_path: Path):
    hdr_path = bin_path.with_suffix(".hdr")
    if not hdr_path.exists():
        hdr_path = bin_path.with_name(bin_path.name + ".hdr")

    meta = envi.read_envi_header(str(hdr_path))

    n_lines = int(meta["lines"])
    n_samples = int(meta["samples"])

    map_info_raw = meta.get("map info", "")
    if isinstance(map_info_raw, list):
        map_info = [str(x).strip() for x in map_info_raw]
    else:
        map_info = [x.strip() for x in map_info_raw.strip("{} ").split(",")]

    x_ul = float(map_info[3])
    y_ul = float(map_info[4])
    x_size = float(map_info[5])
    y_size = float(map_info[6])

    transform = from_origin(x_ul, y_ul, x_size, y_size)

    crs = CRS.from_epsg(32610)
    return transform, crs, n_lines, n_samples


# ---------------------------
# globals for workers
# ---------------------------
_STACK = None
_DATES = None
_QAS = None
_LAM = 20.0
_CONSE = 6
_P_CG = 0.99

_TMASK_B1 = 2
_TMASK_B2 = 11


def _init_worker(stack, dates, qas, lam, conse, p_cg):
    global _STACK, _DATES, _QAS, _LAM, _CONSE, _P_CG

    # IMPORTANT: enforce safe dtypes here
    _STACK = np.ascontiguousarray(stack, dtype=np.float32)
    _DATES = np.asarray(dates, dtype=np.int64)
    _QAS = np.asarray(qas, dtype=np.int16)

    _LAM = lam
    _CONSE = conse
    _P_CG = p_cg


# ---------------------------
# pixel worker
# ---------------------------
def _process_pixel(pixel_idx: int):

    ts = _STACK[:, :, pixel_idx]
    dates = _DATES
    qas = _QAS

    valid = ~np.any(np.isnan(ts), axis=1)
    n_valid = int(valid.sum())

    if n_valid == 0:
        return (pixel_idx, NODATA_INT16, NODATA_INT32, NODATA_FLOAT)

    if n_valid < max(2 * _CONSE, 12):
        return (pixel_idx, NODATA_INT16, NODATA_INT32, NODATA_FLOAT)

    ts_clean = np.ascontiguousarray(ts[valid], dtype=np.float32)
    dates_clean = dates[valid].astype(np.int64)
    qas_clean = qas[valid].astype(np.int16)

    try:
        result = cold_detect_flex(
            dates=dates_clean,
            ts_stack=ts_clean,
            qas=qas_clean,
            lam=_LAM,
            p_cg=_P_CG,
            conse=_CONSE,
            pos=int(pixel_idx + 1),  # CRITICAL FIX
            tmask_b1_index=0, # None,#_TMASK_B1,
            tmask_b2_index=0 None,# _TMASK_B2,
        )
    except Exception as e:
        print(f"[COLD ERROR] pixel={pixel_idx} -> {repr(e)}")
        return (pixel_idx, NODATA_INT16, NODATA_INT32, NODATA_FLOAT)

    if result is None or len(result) == 0:
        return (pixel_idx, 0, 0, 0.0)

    breaks = result[result["t_break"] != 0]
    if len(breaks) == 0:
        return (pixel_idx, 0, 0, 0.0)

    breaks = breaks[np.argsort(breaks["t_break"])]
    first = breaks[0]

    mag = float(np.sqrt(np.mean(first["magnitude"] ** 2)))

    return (
        pixel_idx,
        int(len(breaks)),
        int(first["t_break"]),
        mag,
    )


# ---------------------------
# main
# ---------------------------
def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_workers", type=int, default=None)
    ap.add_argument("--lam", type=float, default=20)
    ap.add_argument("--conse", type=int, default=6)
    ap.add_argument("--p_cg", type=float, default=0.99)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    n_workers = args.n_workers or cpu_count()

    print("Discovering files...")
    files = sorted(input_dir.glob("S2*.bin"))

    file_dates = [(parse_date_from_filename(f), f) for f in files]
    file_dates.sort()

    dates_list = [d for d, _ in file_dates]
    files_list = [f for _, f in file_dates]

    n_times = len(files_list)

    print("Reading metadata...")
    transform, crs, n_lines, n_samples = get_ref_geotransform(files_list[0])

    first = read_envi_bsq(files_list[0])
    n_bands = first.shape[0]
    n_pixels = n_lines * n_samples

    print(f"Stacking {n_times} images...")

    stack = np.empty((n_times, n_bands, n_pixels), dtype=np.float32)
    stack[0] = first.reshape(n_bands, -1)

    for i, f in enumerate(tqdm(files_list[1:]), start=1):
        stack[i] = read_envi_bsq(f).reshape(n_bands, -1)

    qas = np.zeros(n_times, dtype=np.int16)

    dates = np.array([date_to_ordinal(d) for d in dates_list], dtype=np.int64)

    print("Running CCDC...")

    out_breaks = np.full(n_pixels, NODATA_INT16, dtype=np.int16)
    out_first = np.full(n_pixels, NODATA_INT32, dtype=np.int32)
    out_mag = np.full(n_pixels, NODATA_FLOAT, dtype=np.float32)

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(stack, dates, qas, args.lam, args.conse, args.p_cg),
    ) as pool:

        for pid, nb, tb, mg in tqdm(
            pool.imap_unordered(_process_pixel, range(n_pixels), chunksize=256),
            total=n_pixels,
        ):
            out_breaks[pid] = nb
            out_first[pid] = tb
            out_mag[pid] = mg

    print("Writing outputs...")

    profile = dict(
        driver="GTiff",
        height=n_lines,
        width=n_samples,
        crs=crs,
        transform=transform,
        count=1,
        compress="deflate",
    )

    def write(arr, name, dtype, nodata):
        with rasterio.open(output_dir / name, "w", **profile, dtype=dtype, nodata=nodata) as dst:
            dst.write(arr.reshape(n_lines, n_samples).astype(dtype), 1)

    write(out_breaks, "tBreak_count.tif", "int16", NODATA_INT16)
    write(out_first, "tBreak_first.tif", "int32", NODATA_INT32)
    write(out_mag, "MAG_first.tif", "float32", NODATA_FLOAT)

    print("DONE")


if __name__ == "__main__":
    main()



