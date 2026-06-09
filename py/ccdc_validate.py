# -*- coding: utf-8 -*-
"""
ccdc_validate.py
================
Validate CCDC change detection results against reference polygons.

Reads the CCDC output ENVI rasters and a reference polygon file
(GeoJSON, shapefile, or geopackage), rasterizes the polygons onto the
CCDC grid, and reports detection statistics inside vs outside.

Usage
-----
    python ccdc_validate.py \
        --ccdc_dir ./ccdc_out \
        --reference mystery_clearing.geojson \
        --event_year 2023 \
        [--buffer_m 500] \
        [--output_csv validation_stats.csv]

Outputs
-------
  Console summary:
    - Detection rate inside polygons vs background
    - Break date distribution (what fraction of breaks fall in the event year)
    - Magnitude comparison (inside vs outside)
    - Contingency table and accuracy metrics

  Optional CSV with per-polygon statistics.
  Optional rasterized mask saved as reference_mask.bin (ENVI).
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

try:
    from osgeo import gdal, ogr, osr
except ImportError:
    print("ERROR: GDAL/OGR required.  pip install GDAL  or  conda install gdal")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Read ENVI raster via GDAL
# ---------------------------------------------------------------------------
def read_envi(path: Path) -> tuple:
    """Return (array_2d, geotransform, projection_wkt)."""
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open {path}")
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None
    return arr, gt, proj, nodata


# ---------------------------------------------------------------------------
# Rasterize polygons onto the CCDC grid
# ---------------------------------------------------------------------------
def rasterize_reference(ref_path: str, template_path: str,
                        burn_value: int = 1) -> np.ndarray:
    """
    Rasterize a vector file onto the same grid as template_path.
    Returns a 2D uint8 array: burn_value inside polygons, 0 outside.
    Handles CRS reprojection automatically.
    """
    # Open template to get grid specs
    tpl = gdal.Open(template_path, gdal.GA_ReadOnly)
    gt = tpl.GetGeoTransform()
    proj = tpl.GetProjection()
    cols = tpl.RasterXSize
    rows = tpl.RasterYSize
    tpl = None

    # Create in-memory raster
    mem_drv = gdal.GetDriverByName("MEM")
    target_ds = mem_drv.Create("", cols, rows, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(gt)
    target_ds.SetProjection(proj)
    target_ds.GetRasterBand(1).Fill(0)

    # Open vector
    vec_ds = ogr.Open(ref_path)
    if vec_ds is None:
        raise FileNotFoundError(f"Cannot open vector file: {ref_path}")
    src_layer = vec_ds.GetLayer(0)

    # Check if reprojection is needed
    src_srs = src_layer.GetSpatialRef()
    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromWkt(proj)

    if src_srs is not None and not src_srs.IsSame(tgt_srs):
        # Reproject to a temp layer in memory
        mem_ogr_drv = ogr.GetDriverByName("Memory")
        mem_vec_ds = mem_ogr_drv.CreateDataSource("")
        mem_layer = mem_vec_ds.CreateLayer("reprojected", tgt_srs,
                                           ogr.wkbPolygon)
        transform = osr.CoordinateTransformation(src_srs, tgt_srs)

        for feat in src_layer:
            geom = feat.GetGeometryRef().Clone()
            geom.Transform(transform)
            out_feat = ogr.Feature(mem_layer.GetLayerDefn())
            out_feat.SetGeometry(geom)
            mem_layer.CreateFeature(out_feat)

        gdal.RasterizeLayer(target_ds, [1], mem_layer,
                            burn_values=[burn_value])
        mem_vec_ds = None
    else:
        gdal.RasterizeLayer(target_ds, [1], src_layer,
                            burn_values=[burn_value])

    vec_ds = None
    mask = target_ds.GetRasterBand(1).ReadAsArray()
    target_ds = None
    return mask


def create_buffer_mask(ref_path: str, template_path: str,
                       buffer_m: float) -> np.ndarray:
    """
    Create a buffer zone around reference polygons (in the raster CRS).
    Returns mask: 1 = inside buffer but outside polygons, 0 = elsewhere.
    """
    # Open template
    tpl = gdal.Open(template_path, gdal.GA_ReadOnly)
    gt = tpl.GetGeoTransform()
    proj = tpl.GetProjection()
    cols = tpl.RasterXSize
    rows = tpl.RasterYSize
    tpl = None

    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromWkt(proj)

    # Open and reproject vector
    vec_ds = ogr.Open(ref_path)
    src_layer = vec_ds.GetLayer(0)
    src_srs = src_layer.GetSpatialRef()

    mem_ogr_drv = ogr.GetDriverByName("Memory")

    # Reproject + buffer
    buf_ds = mem_ogr_drv.CreateDataSource("")
    buf_layer = buf_ds.CreateLayer("buffered", tgt_srs, ogr.wkbPolygon)

    transform = None
    if src_srs is not None and not src_srs.IsSame(tgt_srs):
        transform = osr.CoordinateTransformation(src_srs, tgt_srs)

    for feat in src_layer:
        geom = feat.GetGeometryRef().Clone()
        if transform:
            geom.Transform(transform)
        buffered = geom.Buffer(buffer_m)
        out_feat = ogr.Feature(buf_layer.GetLayerDefn())
        out_feat.SetGeometry(buffered)
        buf_layer.CreateFeature(out_feat)

    vec_ds = None

    # Rasterize buffered polygons
    mem_drv = gdal.GetDriverByName("MEM")
    buf_rast = mem_drv.Create("", cols, rows, 1, gdal.GDT_Byte)
    buf_rast.SetGeoTransform(gt)
    buf_rast.SetProjection(proj)
    buf_rast.GetRasterBand(1).Fill(0)
    gdal.RasterizeLayer(buf_rast, [1], buf_layer, burn_values=[1])

    buf_mask = buf_rast.GetRasterBand(1).ReadAsArray()
    buf_rast = None
    buf_ds = None

    return buf_mask


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def ordinal_range_for_year(year: int) -> tuple:
    """Return (start_ordinal, end_ordinal) for a calendar year."""
    return date(year, 1, 1).toordinal(), date(year, 12, 31).toordinal()


def analyse_zone(tbreak: np.ndarray, count: np.ndarray,
                 mag_rms: np.ndarray, mask: np.ndarray,
                 nodata_tbreak, nodata_count,
                 event_year: int, zone_name: str) -> dict:
    """Compute statistics for pixels where mask == 1."""
    sel = mask == 1
    n_total = int(sel.sum())
    if n_total == 0:
        print(f"  [{zone_name}] No pixels in zone.")
        return {}

    tb = tbreak[sel]
    ct = count[sel]
    mg = mag_rms[sel]

    # Exclude nodata
    valid = (ct != nodata_count) if nodata_count is not None else np.ones(len(ct), bool)
    n_valid = int(valid.sum())
    n_nodata = n_total - n_valid

    tb_v = tb[valid]
    ct_v = ct[valid].astype(np.float64)
    mg_v = mg[valid]

    # Has any break
    has_break = ct_v > 0
    n_break = int(has_break.sum())
    det_rate = n_break / n_valid if n_valid > 0 else 0.0

    # Break in event year
    yr_start, yr_end = ordinal_range_for_year(event_year)
    in_year = has_break & (tb_v >= yr_start) & (tb_v <= yr_end)
    n_in_year = int(in_year.sum())
    yr_rate = n_in_year / n_valid if n_valid > 0 else 0.0

    # Magnitude stats for pixels with breaks
    mag_break = mg_v[has_break]
    mag_mean = float(np.nanmean(mag_break)) if len(mag_break) > 0 else 0.0
    mag_std = float(np.nanstd(mag_break)) if len(mag_break) > 0 else 0.0

    # Break count stats
    count_mean = float(np.mean(ct_v[has_break])) if n_break > 0 else 0.0

    stats = {
        "zone": zone_name,
        "n_pixels": n_total,
        "n_valid": n_valid,
        "n_nodata": n_nodata,
        "n_any_break": n_break,
        "detection_rate": det_rate,
        f"n_break_in_{event_year}": n_in_year,
        f"detection_rate_{event_year}": yr_rate,
        "mag_rms_mean": mag_mean,
        "mag_rms_std": mag_std,
        "mean_break_count": count_mean,
    }

    print(f"\n  [{zone_name}]")
    print(f"    Pixels:           {n_total:,}  (valid: {n_valid:,}, nodata: {n_nodata:,})")
    print(f"    Any break:        {n_break:,}  ({det_rate*100:.1f}%)")
    print(f"    Break in {event_year}:   {n_in_year:,}  ({yr_rate*100:.1f}%)")
    print(f"    Mean breaks/px:   {count_mean:.2f}")
    print(f"    MAG_rms (breaks): {mag_mean:.2f} +/- {mag_std:.2f}")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Validate CCDC results against reference polygons.")
    ap.add_argument("--ccdc_dir", required=True,
                    help="Directory containing CCDC output .bin files")
    ap.add_argument("--reference", required=True,
                    help="Reference polygon file (geojson, shp, gpkg)")
    ap.add_argument("--event_year", type=int, default=2023,
                    help="Year the event occurred (default: 2023)")
    ap.add_argument("--buffer_m", type=float, default=500,
                    help="Buffer distance in metres for background zone (default: 500)")
    ap.add_argument("--output_csv", default=None,
                    help="Optional CSV output for per-zone stats")
    ap.add_argument("--save_mask", action="store_true",
                    help="Save the rasterized reference mask as ENVI .bin")
    args = ap.parse_args()

    ccdc_dir = Path(args.ccdc_dir)
    ref_path = args.reference

    # ------------------------------------------------------------------
    # 1. Load CCDC outputs
    # ------------------------------------------------------------------
    print("Loading CCDC outputs ...")

    tbreak_path = ccdc_dir / "tBreak_first.bin"
    count_path  = ccdc_dir / "tBreak_count.bin"
    magrms_path = ccdc_dir / "MAG_rms.bin"

    for p in [tbreak_path, count_path, magrms_path]:
        if not p.exists():
            print(f"ERROR: {p} not found")
            sys.exit(1)

    tbreak, gt, proj, nd_tb = read_envi(tbreak_path)
    count, _, _, nd_ct      = read_envi(count_path)
    mag_rms, _, _, _        = read_envi(magrms_path)

    print(f"  Grid: {tbreak.shape[0]} x {tbreak.shape[1]}")
    print(f"  tBreak nodata: {nd_tb}   count nodata: {nd_ct}")

    # Load per-band magnitudes if available
    band_names = ["B12", "B11", "B9", "B8"]
    mag_bands = {}
    for bname in band_names:
        bp = ccdc_dir / f"MAG_{bname}.bin"
        if bp.exists():
            mag_bands[bname], _, _, _ = read_envi(bp)

    # ------------------------------------------------------------------
    # 2. Rasterize reference polygons
    # ------------------------------------------------------------------
    print(f"\nRasterizing reference: {ref_path} ...")
    ref_mask = rasterize_reference(ref_path, str(tbreak_path))
    n_ref = int((ref_mask == 1).sum())
    print(f"  Reference pixels: {n_ref:,}")

    if n_ref == 0:
        print("ERROR: no pixels fell inside reference polygons. "
              "Check CRS alignment.")
        sys.exit(1)

    # Background: buffer minus reference
    print(f"  Creating {args.buffer_m}m buffer background zone ...")
    buf_mask = create_buffer_mask(ref_path, str(tbreak_path), args.buffer_m)
    bg_mask = ((buf_mask == 1) & (ref_mask == 0)).astype(np.uint8)
    n_bg = int((bg_mask == 1).sum())
    print(f"  Background pixels: {n_bg:,}")

    # Save mask if requested
    if args.save_mask:
        out_mask_path = ccdc_dir / "reference_mask.bin"
        drv = gdal.GetDriverByName("ENVI")
        ds = drv.Create(str(out_mask_path), ref_mask.shape[1],
                        ref_mask.shape[0], 1, gdal.GDT_Byte,
                        options=["INTERLEAVE=BSQ"])
        ds.SetGeoTransform(gt)
        ds.SetProjection(proj)
        ds.GetRasterBand(1).WriteArray(ref_mask)
        ds.FlushCache()
        ds = None
        print(f"  Saved mask -> {out_mask_path}")

    # ------------------------------------------------------------------
    # 3. Analyse inside vs outside
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  CCDC VALIDATION RESULTS")
    print("=" * 60)

    stats_inside = analyse_zone(tbreak, count, mag_rms, ref_mask,
                                nd_tb, nd_ct, args.event_year,
                                "INSIDE reference polygons")

    stats_outside = analyse_zone(tbreak, count, mag_rms, bg_mask,
                                 nd_tb, nd_ct, args.event_year,
                                 f"BACKGROUND ({args.buffer_m}m buffer)")

    # ------------------------------------------------------------------
    # 4. Comparative summary
    # ------------------------------------------------------------------
    if stats_inside and stats_outside:
        yr = args.event_year
        ri = stats_inside.get(f"detection_rate_{yr}", 0)
        ro = stats_outside.get(f"detection_rate_{yr}", 0)
        ratio = ri / ro if ro > 0 else float("inf")

        print(f"\n  --- COMPARISON ---")
        print(f"    Detection rate inside  ({yr}): {ri*100:.1f}%")
        print(f"    Detection rate outside ({yr}): {ro*100:.1f}%")
        print(f"    Ratio (inside/outside):         {ratio:.1f}x")

        mi = stats_inside.get("mag_rms_mean", 0)
        mo = stats_outside.get("mag_rms_mean", 0)
        print(f"    Mean MAG_rms inside:            {mi:.2f}")
        print(f"    Mean MAG_rms outside:           {mo:.2f}")

        if ratio > 2.0:
            print(f"\n    >> CCDC captures the {yr} event: inside detection")
            print(f"       rate is {ratio:.1f}x higher than background.")
        elif ratio > 1.2:
            print(f"\n    >> Weak detection: inside rate only {ratio:.1f}x background.")
            print(f"       Consider tuning --consecutive or --alpha.")
        else:
            print(f"\n    >> Event NOT clearly captured by CCDC.")
            print(f"       Inside/outside rates are similar ({ratio:.1f}x).")

    # Per-band magnitude comparison inside polygons
    if mag_bands and stats_inside:
        print(f"\n  --- PER-BAND MAGNITUDE (inside, pixels with breaks) ---")
        sel_in = (ref_mask == 1) & (count > 0)
        for bname, marr in mag_bands.items():
            vals = marr[sel_in]
            if len(vals) > 0:
                vals = vals[~np.isnan(vals)]
                print(f"    {bname}: mean={np.mean(vals):.2f}  "
                      f"std={np.std(vals):.2f}  "
                      f"median={np.median(vals):.2f}")

    # Break date distribution inside polygons
    if stats_inside:
        sel_in = (ref_mask == 1) & (count > 0)
        tb_in = tbreak[sel_in]
        if len(tb_in) > 0:
            print(f"\n  --- BREAK DATE DISTRIBUTION (inside polygons) ---")
            for yr in range(args.event_year - 2, args.event_year + 3):
                yr_s, yr_e = ordinal_range_for_year(yr)
                n_yr = int(((tb_in >= yr_s) & (tb_in <= yr_e)).sum())
                pct = 100.0 * n_yr / len(tb_in)
                bar = "#" * int(pct / 2)
                print(f"    {yr}: {n_yr:6,}  ({pct:5.1f}%)  {bar}")

    print("\n" + "=" * 60)

    # ------------------------------------------------------------------
    # 5. Optional CSV
    # ------------------------------------------------------------------
    if args.output_csv:
        import csv
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(stats_inside.keys()))
            w.writeheader()
            w.writerow(stats_inside)
            if stats_outside:
                w.writerow(stats_outside)
        print(f"\nStats saved to {args.output_csv}")


if __name__ == "__main__":
    main()
