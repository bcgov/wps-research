"""
Extract a binary mask from SCL ENVI files where pixel == any target_value.
Output is a Float32 raster: 1 where match, 0 elsewhere, NaN where nodata.
Processes files in parallel using ThreadPoolExecutor (I/O-bound workload).
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import cpu_count
from osgeo import gdal
import numpy as np


def _process_file(args: tuple):
    f, values, out_dir = args
    src = gdal.Open(str(f))
    if src is None:
        print(f"Skipping {f.name} — could not open.")
        return

    data = src.GetRasterBand(1).ReadAsArray().astype(np.float32)

    # Match any of the target values
    match = np.isin(data, values)
    mask = match.astype(np.float32)

    # Preserve nodata (SCL == 0) unless 0 is one of the target values
    if 0 not in values:
        mask[data == 0] = np.nan

    label = "_".join(str(v) for v in sorted(values))
    out_name = Path(out_dir) / f.name.replace(".bin", f"_scl_mask_val{label}.bin")

    driver = gdal.GetDriverByName("ENVI")
    ds = driver.Create(
        str(out_name),
        src.RasterXSize,
        src.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=["INTERLEAVE=BSQ"]
    )
    ds.SetGeoTransform(src.GetGeoTransform())
    ds.SetProjection(src.GetProjection())
    band = ds.GetRasterBand(1)
    band.WriteArray(mask)
    band.SetNoDataValue(np.nan)
    band.SetDescription(f"SCL values {label}")
    ds.SetMetadata({"band names": f"SCL values {label}"}, "ENVI")
    ds.FlushCache()
    ds = None
    src = None
    print(f"Written: {out_name}")


def extract_scl_mask(scl_dir: str, values: list[int], out_dir: str):
    scl_path = Path(scl_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(scl_path.glob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"No .bin files found in {scl_dir}")

    n_workers = cpu_count()
    print(f"Processing {len(bin_files)} file(s) across {n_workers} workers — values: {values}")

    tasks = [(f, values, out_path) for f in bin_files]
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_file, t): t[0].name for t in tasks}
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                print(f"Error processing {futures[future]}: {exc}")

    print("All done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract binary mask from SCL ENVI files.")
    parser.add_argument("scl_dir", type=str, help="Directory containing SCL .bin files")
    parser.add_argument(
        "value",
        type=lambda s: [int(v) for v in s.split(",")],
        help="SCL value(s) to mask — single int or comma-separated list (e.g. 0,1,4)"
    )
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    extract_scl_mask(args.scl_dir, args.value, args.outdir)