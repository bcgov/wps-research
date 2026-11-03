#!/usr/bin/env python3
"""
20261102 t3_ratio_scan.py -- find the pre/post T3 pair whose post/pre ratio gives the
best contrast for a set of polygons (e.g. fire perimeters), then write that
ratio image plus a PNG with the polygons drawn on it.

Usage (run inside the folder that holds the T3 folders, e.g. the warp folder):
    python t3_ratio_scan.py prot_current_fire_polys.shp

What it does
  1. Finds every subfolder of the present working directory (NOT recursive)
     that holds T11, T22 and T33 (.bin + .hdr), and takes its date (YYYYMMDD)
     from the folder name, e.g. 20260705_ALOS4_quad -> 2026-07-05.
  2. Checks the folders share one grid (samples, lines, map info,
     geotransform, CRS, all read with GDAL). Folders that don't match the
     majority grid are skipped with a warning.
  3. Rasterizes the union of all polygons onto the grid ("inside"), and a
     ring of background around them ("outside" = pixels between --gap and
     --gap + --ring metres from the nearest polygon). The shapefile can be in
     any CRS; it is reprojected to the raster CRS.
  4. For every pair with pre date < post date, takes the 3-band log ratio
     log10(post/pre) of T22, T33, T11 and scores it:
         score = JM x coverage
       JM       = Jeffries-Matusita distance (0..2) between the inside and
                  outside distributions of the 3-band log ratio (Gaussian
                  model: JM = 2(1 - exp(-B)), B = Bhattacharyya distance;
                  JM = 2 x squared Hellinger distance). 0 = inside and
                  outside look the same, 2 = completely separable. Direction
                  free: a drop (e.g. loss of volume scattering in T33 after
                  fire) counts the same as a rise.
       coverage = fraction of polygon pixels on the grid where both pre and
                  post have data (polygon area with no data counts as zero
                  contrast).
     Values are winsorized at the 0.5/99.5 percentiles before scoring, and
     at most --max-samples pixels per class are used (fixed random seed, so
     re-runs give identical results).
  5. Writes to the present working directory:
       ratio.bin / ratio.hdr  best pair, 3-band float32 ENVI BSQ, bands
                              T22, T33, T11 post/pre (same output as
                              t3_ratio.py for that pair); header map info /
                              CRS copied from the inputs
       ratio.png              RGB = T22, T33, T11 ratio in dB, per-band 2-98%
                              stretch; polygons outlined in ONE colour picked
                              from the image data for maximum contrast (see
                              pick_line_colour); title names the pre/post
                              folders and dates
       ratio_scan.csv         scores for every pair, best first

Pixels where pre <= 0 or either value is not finite are NaN in ratio.bin.
"""
import argparse
import csv
import datetime
import os
import re
import struct
import subprocess
import sys
import warnings

import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

ORDER = ["T22", "T33", "T11"]   # output band order = R, G, B in the PNG
SEED = 0                        # fixed seed -> identical results on re-run

# header keys describing the data layout -- rewritten for the output; all
# other keys (map info, projection info, coordinate system string, ...) are
# copied from the pre T11.hdr
LAYOUT_KEYS = {"samples", "lines", "bands", "header offset", "file type",
               "data type", "interleave", "byte order", "band names",
               "description", "data ignore value", "wavelength", "fwhm",
               "default bands", "bbl", "data gain values",
               "data offset values", "statistics"}
BRACE_KEYS = {"map info", "projection info", "coordinate system string",
              "band names", "description", "wavelength", "fwhm",
              "default bands", "bbl", "data gain values",
              "data offset values"}

DATE_RE = re.compile(r"(?<!\d)((?:19|20)\d\d)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")


def err(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(1)


def warn(msg):
    print("WARNING: " + msg, file=sys.stderr)


def import_matplotlib():
    """Import matplotlib (headless); install it into this Python if missing."""
    try:
        import matplotlib
    except ImportError:
        print("matplotlib not found -- installing it into this Python ...")
        for extra in (["--break-system-packages"], []):
            if subprocess.call([sys.executable, "-m", "pip", "install",
                                "--quiet"] + extra + ["matplotlib"]) == 0:
                break
        import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    return plt, LineCollection


# ----------------------------------------------------------------- ENVI / grid
def norm(s):
    """Normalize an ENVI {...} value for comparison: drop braces & whitespace."""
    if s is None:
        return None
    return re.sub(r"\s+", "", s.strip().strip("{}"))


def envi_header(ds):
    """ENVI header (key, value) pairs as parsed by GDAL (metadata domain 'ENVI').
    GDAL stores keys with spaces replaced by '_'; convert back."""
    out = []
    for item in ds.GetMetadata_List("ENVI") or []:
        k, _, v = item.partition("=")
        out.append((k.strip().replace("_", " ").lower(), v.strip()))
    return out


def fmt_value(key, val):
    if key in BRACE_KEYS and not val.startswith("{"):
        return "{" + val + "}"
    return val


def srs_from_wkt(wkt):
    if not wkt:
        return None
    s = osr.SpatialReference()
    s.ImportFromWkt(wkt)
    if hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return s


def same_crs(wkt_a, wkt_b):
    if not wkt_a and not wkt_b:
        return True
    if not wkt_a or not wkt_b:
        return False
    return bool(srs_from_wkt(wkt_a).IsSame(srs_from_wkt(wkt_b)))


def open_grid(folder, name):
    ds = gdal.Open(os.path.join(folder, name + ".bin"), gdal.GA_ReadOnly)
    if ds.GetDriver().ShortName != "ENVI":
        err("GDAL did not open %s/%s.bin with the ENVI driver" % (folder, name))
    return {"ds": ds, "samples": ds.RasterXSize, "lines": ds.RasterYSize,
            "bands": ds.RasterCount,
            "gt": ds.GetGeoTransform(can_return_null=True),
            "wkt": ds.GetProjection(),
            "map_info": dict(envi_header(ds)).get("map info")}


def same_grid(a, b):
    if a["bands"] != 1 or b["bands"] != 1:
        return False
    if (a["samples"], a["lines"]) != (b["samples"], b["lines"]):
        return False
    if norm(a["map_info"]) != norm(b["map_info"]):
        return False
    if (a["gt"] is None) != (b["gt"] is None):
        return False
    if a["gt"] is not None and not np.allclose(a["gt"], b["gt"], rtol=0, atol=1e-6):
        return False
    return same_crs(a["wkt"], b["wkt"])


def find_t3_dirs():
    """Subfolders of the present working directory (not recursive) holding
    T11/T22/T33 .bin + .hdr, with a YYYYMMDD date in the folder name."""
    found = []
    for e in sorted(os.scandir("."), key=lambda e: e.name):
        if not e.is_dir():
            continue
        if not all(os.path.isfile(os.path.join(e.name, b + ext))
                   for b in ORDER for ext in (".bin", ".hdr")):
            continue
        m = DATE_RE.search(e.name)
        if not m:
            warn("skipping %s: no YYYYMMDD date in the folder name" % e.name)
            continue
        found.append((datetime.date(int(m.group(1)), int(m.group(2)),
                                    int(m.group(3))), e.name))
    found.sort()
    return found


# -------------------------------------------------------------------- polygons
def polygon_parts(g):
    t = ogr.GT_Flatten(g.GetGeometryType())
    if t == ogr.wkbPolygon:
        yield g
    elif t in (ogr.wkbMultiPolygon, ogr.wkbGeometryCollection):
        for i in range(g.GetGeometryCount()):
            yield from polygon_parts(g.GetGeometryRef(i))


def load_polygons(shp, dst_srs, extent):
    """Polygons reprojected to the raster CRS; keeps those overlapping extent."""
    try:
        ds = ogr.Open(shp)
    except Exception:
        ds = None
    if ds is None:
        err("could not open shapefile " + os.path.basename(shp))
    lyr = ds.GetLayer(0)
    src_srs = lyr.GetSpatialRef()
    ct = None
    if src_srs is None:
        warn("%s has no CRS (.prj); assuming it matches the rasters"
             % os.path.basename(shp))
    elif dst_srs is None:
        warn("rasters have no CRS; using polygon coordinates as they are")
    else:
        src = src_srs.Clone()
        if hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
            src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not src.IsSame(dst_srs):
            ct = osr.CoordinateTransformation(src, dst_srs)
    xmin, xmax, ymin, ymax = extent
    polys, n_poly_feat, n_on = [], 0, 0
    for f in lyr:
        g = f.GetGeometryRef()
        if g is None:
            continue
        g = g.Clone()
        if g.HasCurveGeometry():
            g = g.GetLinearGeometry()
        parts = [p.Clone() for p in polygon_parts(g)]
        if not parts:
            continue
        n_poly_feat += 1
        on = False
        for p in parts:
            if ct is not None:
                p.Transform(ct)
            p.FlattenTo2D()
            ex0, ex1, ey0, ey1 = p.GetEnvelope()
            if ex1 < xmin or ex0 > xmax or ey1 < ymin or ey0 > ymax:
                continue
            polys.append(p)
            on = True
        n_on += on
    return polys, n_poly_feat, n_on


def rings_xy(polys):
    out = []
    for p in polys:
        for i in range(p.GetGeometryCount()):
            pts = p.GetGeometryRef(i).GetPoints()
            if pts and len(pts) >= 2:
                out.append(np.asarray(pts, dtype=np.float64)[:, :2])
    return out


def line_geom(xy):
    """OGR LineString from an (N,2) array (via WKB, fast)."""
    wkb = struct.pack("<BII", 1, 2, len(xy)) + np.ascontiguousarray(xy, "<f8").tobytes()
    return ogr.CreateGeometryFromWkb(wkb)


def mem_layer(srs, geoms):
    drv = ogr.GetDriverByName("Memory") or ogr.GetDriverByName("MEM")
    mds = drv.CreateDataSource("scan")
    lyr = mds.CreateLayer("g", srs=srs, geom_type=ogr.wkbUnknown)
    defn = lyr.GetLayerDefn()
    for g in geoms:
        feat = ogr.Feature(defn)
        feat.SetGeometry(g)
        lyr.CreateFeature(feat)
    return mds, lyr          # keep mds alive while lyr is used


def burn(lyr, nx, ny, gt, wkt, all_touched=False):
    ds = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(gt)
    if wkt:
        ds.SetProjection(wkt)
    gdal.RasterizeLayer(ds, [1], lyr, burn_values=[1],
                        options=["ALL_TOUCHED=TRUE"] if all_touched else [])
    return ds


def distance_to_polygons(mask_ds, maxdist):
    """Distance (map units) from each pixel to the nearest polygon pixel;
    -1 where farther than maxdist."""
    nx, ny = mask_ds.RasterXSize, mask_ds.RasterYSize
    dds = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Float32)
    dds.SetGeoTransform(mask_ds.GetGeoTransform())
    dds.SetProjection(mask_ds.GetProjection())
    gdal.ComputeProximity(mask_ds.GetRasterBand(1), dds.GetRasterBand(1),
                          ["VALUES=1", "DISTUNITS=GEO",
                           "MAXDIST=%.6f" % maxdist, "NODATA=-1"])
    return dds.GetRasterBand(1).ReadAsArray()


# --------------------------------------------------------------------- scoring
def bhattacharyya(a, b):
    """Gaussian Bhattacharyya distance between samples a, b (rows = pixels);
    also returns the Mahalanobis distance between the class means."""
    m1, m2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    k = s1.shape[0]
    ridge = 1e-6 * max((np.trace(s1) + np.trace(s2)) / (2 * k), 1e-30)
    s1 = s1 + ridge * np.eye(k)
    s2 = s2 + ridge * np.eye(k)
    s = 0.5 * (s1 + s2)
    dm = m1 - m2
    maha2 = float(dm @ np.linalg.solve(s, dm))
    ld = np.linalg.slogdet(s)[1]
    ld1 = np.linalg.slogdet(s1)[1]
    ld2 = np.linalg.slogdet(s2)[1]
    return 0.125 * maha2 + 0.5 * (ld - 0.5 * (ld1 + ld2)), np.sqrt(maha2)


def pair_metrics(lpre, lpost, n_in, min_px):
    """lpre/lpost: (pixels, 3) log10 T22,T33,T11 at the sampled pixels (first
    n_in rows = inside polygons, rest = background); NaN = no data."""
    lr = lpost - lpre
    ok = ~np.isnan(lr).any(axis=1)
    a = lr[:n_in][ok[:n_in]].astype(np.float64)
    b = lr[n_in:][ok[n_in:]].astype(np.float64)
    r = {"n_in": len(a), "n_out": len(b), "coverage": float(ok[:n_in].mean()),
         "jm": np.nan, "bhatt": np.nan, "maha": np.nan, "score": np.nan}
    for band in ORDER:
        r["db_" + band] = np.nan
    if len(a) < min_px or len(b) < min_px:
        return r
    dmed = 10.0 * (np.median(a, 0) - np.median(b, 0))     # dB, inside - outside
    lo, hi = np.percentile(np.vstack([a, b]), [0.5, 99.5], axis=0)
    B, D = bhattacharyya(np.clip(a, lo, hi), np.clip(b, lo, hi))
    jm = 2.0 * (1.0 - np.exp(-B))
    r.update(jm=jm, bhatt=B, maha=D, score=jm * r["coverage"])
    for i, band in enumerate(ORDER):
        r["db_" + band] = dmed[i]
    return r


# ----------------------------------------------------------------- PNG helpers
def block_mean(a, k):
    if k == 1:
        return a
    ny, nx = a.shape[0] // k, a.shape[1] // k
    a = a[:ny * k, :nx * k].reshape(ny, k, nx, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=(1, 3))


def dilate(m, r):
    """Binary dilation with a (2r+1)x(2r+1) square, no wrap-around."""
    for axis in (0, 1):
        n = m.shape[axis]
        out = m.copy()
        for d in range(1, r + 1):
            if d >= n:
                break
            src = [slice(None), slice(None)]
            dst = [slice(None), slice(None)]
            src[axis], dst[axis] = slice(d, n), slice(0, n - d)
            out[tuple(dst)] |= m[tuple(src)]
            src[axis], dst[axis] = slice(0, n - d), slice(d, n)
            out[tuple(dst)] |= m[tuple(src)]
        m = out
    return m


def srgb_to_lab(rgb):
    """sRGB in [0,1] (..., 3) -> CIELAB (D65)."""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = (c @ M.T) / np.array([0.95047, 1.0, 1.08883])
    e = (6.0 / 29.0) ** 3
    f = np.where(xyz > e, np.cbrt(xyz), xyz / (3 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def pick_line_colour(pixels, rng, n_samples=20000, kL=2.0, pct=5):
    """Pick the one colour that stands out most against `pixels` (the image
    colours under and next to the polygon outlines).
    Candidates: 17x17x17 grid over the RGB cube. Distance: CIELAB with the
    lightness difference weighted x2 (thin lines are seen mostly through
    lightness contrast). Objective: the 5th percentile of the distance over
    the pixels, i.e. the line contrasts with ~95% of what it is drawn over;
    the candidate with the largest value wins."""
    if len(pixels) > n_samples:
        pixels = pixels[rng.choice(len(pixels), n_samples, replace=False)]
    lab_p = srgb_to_lab(pixels.astype(np.float64))
    lv = np.linspace(0.0, 1.0, 17)
    cand = np.stack(np.meshgrid(lv, lv, lv, indexing="ij"), -1).reshape(-1, 3)
    lab_c = srgb_to_lab(cand)
    w = np.array([kL, 1.0, 1.0])
    score = np.empty(len(cand))
    for i in range(0, len(cand), 64):
        d = np.sqrt((((lab_c[i:i + 64, None, :] - lab_p[None, :, :]) * w) ** 2).sum(-1))
        score[i:i + 64] = np.percentile(d, pct, axis=1)
    j = int(np.argmax(score))
    return cand[j], score[j]


def make_png(path, rgb, extent, rings, colour, lw_px, title_lines):
    plt, LineCollection = import_matplotlib()
    H, W = rgb.shape[:2]
    dpi = 100
    maxchars = max(len(s) for s in title_lines)
    fs = float(np.clip(0.95 * W * 72.0 / (0.6 * maxchars * dpi), 7, 18))
    line_px = fs * 1.45 * dpi / 72.0
    top = int(np.ceil(line_px * (len(title_lines) + 0.6)))
    fig = plt.figure(figsize=(W / dpi, (H + top) / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, H / float(H + top)])
    ax.imshow(rgb, extent=extent, interpolation="nearest", aspect="auto")
    ax.add_collection(LineCollection(rings, colors=[tuple(colour)],
                                     linewidths=lw_px * 72.0 / dpi))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_axis_off()
    for i, s in enumerate(title_lines):
        y = 1.0 - (0.3 + i + 0.5) * line_px / (H + top)
        fig.text(0.5, y, s, ha="center", va="center", fontsize=fs)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(
        description="Scan all T3 folders in the present working directory for "
                    "the pre/post pair whose post/pre ratio (T22, T33, T11) best "
                    "contrasts the given polygons; write ratio.bin/.hdr, "
                    "ratio.png and ratio_scan.csv.")
    ap.add_argument("shapefile", help="polygon shapefile (any CRS)")
    ap.add_argument("--ring", type=float, default=1000.0,
                    help="width (m) of the background ring around the polygons; "
                         "0 = all non-polygon pixels (default 1000)")
    ap.add_argument("--gap", type=float, default=60.0,
                    help="gap (m) between polygon edge and background ring, "
                         "skips edge-mixed pixels (default 60)")
    ap.add_argument("--max-samples", type=int, default=300000,
                    help="max pixels sampled per class for scoring (default 300000)")
    ap.add_argument("--min-pixels", type=int, default=100,
                    help="min valid pixels per class to score a pair (default 100)")
    ap.add_argument("--png-max", type=int, default=2500,
                    help="max PNG width/height in pixels; larger rasters are "
                         "block-averaged for display (default 2500)")
    ap.add_argument("--pair", nargs=2, metavar=("PRE", "POST"),
                    help="render this pair instead of the best one "
                         "(the scan table is still written)")
    a = ap.parse_args()
    shp_name = os.path.basename(a.shapefile)
    rng = np.random.default_rng(SEED)

    # ---- 1. T3 folders ------------------------------------------------------
    dirs = find_t3_dirs()
    if len(dirs) < 2:
        err("need at least 2 dated T3 folders in the present working directory "
            "(found %d)" % len(dirs))
    dates = {n: d for d, n in dirs}

    # ---- 2. common grid -----------------------------------------------------
    grids = {}
    for d, n in dirs:
        g = {b: open_grid(n, b) for b in ORDER}
        bad = [b for b in ORDER if not same_grid(g["T11"], g[b])]
        if bad:
            warn("skipping %s: %s grid differs from T11 in the same folder"
                 % (n, ", ".join(bad)))
            continue
        grids[n] = g
    groups = []
    for n in grids:
        for grp in groups:
            if same_grid(grids[grp[0]]["T11"], grids[n]["T11"]):
                grp.append(n)
                break
        else:
            groups.append([n])
    if not groups:
        err("no usable T3 folders")
    groups.sort(key=len, reverse=True)
    for grp in groups[1:]:
        for n in grp:
            warn("skipping %s: grid (size / map info / geotransform / CRS) "
                 "differs from %s" % (n, groups[0][0]))
    names = sorted(groups[0], key=lambda n: (dates[n], n))
    if len(names) < 2:
        err("need at least 2 T3 folders on the same grid")
    ref = grids[names[0]]["T11"]
    nx, ny, gt, wkt = ref["samples"], ref["lines"], ref["gt"], ref["wkt"]
    if gt is None:
        err("%s/T11.hdr has no map info / geotransform" % names[0])
    print("T3 folders on a common grid (%d samples x %d lines):" % (nx, ny))
    for n in names:
        print("  %s  %s" % (dates[n].isoformat(), n))
    print("map info: %s" % ref["map_info"])

    # ---- 3. polygons -> inside / background masks ---------------------------
    xs = (gt[0], gt[0] + nx * gt[1])
    ys = (gt[3], gt[3] + ny * gt[5])
    dst_srs = srs_from_wkt(wkt)
    polys, n_poly_feat, n_on = load_polygons(
        a.shapefile, dst_srs, (min(xs), max(xs), min(ys), max(ys)))
    if n_poly_feat == 0:
        err("no polygon geometries in " + shp_name)
    if not polys:
        err("none of the %d polygons in %s overlap the raster grid"
            % (n_poly_feat, shp_name))
    pds, plyr = mem_layer(dst_srs, polys)
    mask_ds = burn(plyr, nx, ny, gt, wkt)
    inside = mask_ds.GetRasterBand(1).ReadAsArray().astype(bool)
    if not inside.any():
        err("the polygons in %s cover no pixel centres on the grid" % shp_name)
    maxd = a.gap + max(a.ring, 0.0)
    if maxd > 0:
        dist = distance_to_polygons(mask_ds, maxd)
        if a.ring > 0:
            outside = (dist > a.gap) & (dist <= maxd)
        else:
            outside = ~inside & ((dist > a.gap) | (dist < 0))
        del dist
    else:
        outside = ~inside
    if not outside.any():
        err("no background pixels around the polygons; try a larger --ring")
    px_ha = abs(gt[1] * gt[5]) / 1e4
    bg_desc = ("%g-%g m ring" % (a.gap, maxd)) if a.ring > 0 else \
              ("all pixels > %g m from polygons" % a.gap)
    print("polygons: %d in %s, %d on the grid; inside %d px (%.0f ha), "
          "background (%s) %d px"
          % (n_poly_feat, shp_name, n_on, inside.sum(), inside.sum() * px_ha,
             bg_desc, outside.sum()))

    # ---- 4. sample pixels, cache log10 values per folder --------------------
    in_idx = np.flatnonzero(inside.ravel())
    out_idx = np.flatnonzero(outside.ravel())
    del outside
    if in_idx.size > a.max_samples:
        in_idx = np.sort(rng.choice(in_idx, a.max_samples, replace=False))
    if out_idx.size > a.max_samples:
        out_idx = np.sort(rng.choice(out_idx, a.max_samples, replace=False))
    idx = np.concatenate([in_idx, out_idx])
    n_in = in_idx.size
    print("reading %d folders (%d inside + %d background samples) ..."
          % (len(names), n_in, out_idx.size))
    logs = {}
    for n in names:
        v = np.empty((idx.size, 3), np.float32)
        for i, b in enumerate(ORDER):
            v[:, i] = grids[n][b]["ds"].GetRasterBand(1).ReadAsArray().ravel()[idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            good = np.all(np.isfinite(v) & (v > 0), axis=1)
        lv = np.full(v.shape, np.nan, np.float32)
        lv[good] = np.log10(v[good])
        logs[n] = lv
        print("  %-28s data at %5.1f%% of polygon pixels"
              % (n, 100.0 * good[:n_in].mean()))

    # ---- 5. score every pre < post pair --------------------------------------
    results = []
    for i, p in enumerate(names):
        for q in names[i + 1:]:
            if dates[p] < dates[q]:
                r = pair_metrics(logs[p], logs[q], n_in, a.min_pixels)
                r.update(pre=p, post=q, days=(dates[q] - dates[p]).days)
                results.append(r)
    if not results:
        err("no pre/post pairs (all folders have the same date?)")
    results.sort(key=lambda r: -r["score"] if np.isfinite(r["score"]) else np.inf)
    for k, r in enumerate(results, 1):
        r["rank"] = k

    with open("ratio_scan.csv", "w", newline="") as f:
        cols = ["rank", "pre", "post", "days", "score", "jm", "coverage",
                "bhatt", "maha", "db_T22", "db_T33", "db_T11", "n_in", "n_out"]
        w = csv.writer(f)
        w.writerow(["rank", "pre", "post", "days", "score", "JM", "coverage",
                    "bhattacharyya", "mahalanobis",
                    "dB_T22_in_minus_out", "dB_T33_in_minus_out",
                    "dB_T11_in_minus_out", "n_inside", "n_background"])
        for r in results:
            w.writerow([("%.6g" % r[c]) if isinstance(r[c], float) else r[c]
                        for c in cols])

    print("\nscore = JM x coverage;  dB columns = median 10log10(post/pre), "
          "inside minus background")
    print("%4s  %-24s %-24s %5s %6s %6s %6s %7s %7s %7s"
          % ("rank", "pre", "post", "days", "score", "JM", "cover",
             "dB T22", "dB T33", "dB T11"))
    for r in results[:15]:
        print("%4d  %-24s %-24s %5d %6.3f %6.3f %5.0f%% %7.2f %7.2f %7.2f"
              % (r["rank"], r["pre"], r["post"], r["days"], r["score"], r["jm"],
                 100 * r["coverage"], r["db_T22"], r["db_T33"], r["db_T11"]))
    if len(results) > 15:
        print("  ... %d more in ratio_scan.csv" % (len(results) - 15))

    # ---- 6. chosen pair -------------------------------------------------------
    if a.pair:
        pre, post = [os.path.basename(os.path.normpath(x)) for x in a.pair]
        for x in (pre, post):
            if x not in names:
                err("--pair: %s is not one of the T3 folders on the common grid" % x)
        if dates[pre] >= dates[post]:
            err("--pair: pre (%s) must be earlier than post (%s)" % (pre, post))
        best = [r for r in results if r["pre"] == pre and r["post"] == post][0]
    else:
        best = results[0]
        if not np.isfinite(best["score"]):
            err("no pair has >= %d valid pixels both inside and around the "
                "polygons" % a.min_pixels)
    pre, post = best["pre"], best["post"]
    print("\nselected: pre %s  post %s  (JM %.3f, coverage %.0f%%, score %.3f)"
          % (pre, post, best["jm"], 100 * best["coverage"], best["score"]))

    # ---- 7. ratio.bin / ratio.hdr ---------------------------------------------
    k = max(1, int(np.ceil(max(nx, ny) / float(a.png_max))))
    disp = []
    with open("ratio.bin", "wb") as f:
        for b in ORDER:
            p0 = grids[pre][b]["ds"].GetRasterBand(1).ReadAsArray().astype(np.float32)
            p1 = grids[post][b]["ds"].GetRasterBand(1).ReadAsArray().astype(np.float32)
            r = np.full(p0.shape, np.nan, dtype=np.float32)
            with np.errstate(invalid="ignore"):
                ok = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0)
            np.divide(p1, p0, out=r, where=ok)
            r.astype("<f4").tofile(f)
            with np.errstate(divide="ignore", invalid="ignore"):
                db = 10.0 * np.log10(r)
            db[~np.isfinite(db)] = np.nan
            disp.append(block_mean(db, k).astype(np.float32))
            del p0, p1, r, ok, db
    if os.path.exists("ratio.bin.aux.xml"):     # stale GDAL stats from an old ratio.bin
        os.remove("ratio.bin.aux.xml")

    band_names = ["%s post/pre (%s / %s)" % (b, post, pre) for b in ORDER]
    lines = [
        "ENVI",
        "description = {T3 diagonal ratio post/pre. pre=%s post=%s. selected by "
        "t3_ratio_scan.py for %s: JM=%.4f coverage=%.4f}"
        % (pre, post, shp_name, best["jm"], best["coverage"]),
        "samples = %d" % nx,
        "lines = %d" % ny,
        "bands = %d" % len(ORDER),
        "header offset = 0",
        "file type = ENVI Standard",
        "data type = 4",
        "interleave = bsq",
        "byte order = 0",
    ]
    for key, val in envi_header(grids[pre]["T11"]["ds"]):
        if key not in LAYOUT_KEYS:
            lines.append("%s = %s" % (key, fmt_value(key, val)))
    lines.append("band names = {" + ",\n".join(band_names) + "}")
    with open("ratio.hdr", "w") as f:
        f.write("\n".join(lines) + "\n")

    chk = gdal.Open("ratio.bin")
    if (chk.RasterXSize, chk.RasterYSize, chk.RasterCount) != (nx, ny, 3) or \
            norm(dict(envi_header(chk)).get("map info")) != norm(ref["map_info"]) or \
            not same_crs(chk.GetProjection(), wkt):
        err("ratio.bin read-back does not match the input grid")
    chk = None
    print("wrote ratio.bin, ratio.hdr  (bands: %s)" % ", ".join(band_names))

    # ---- 8. ratio.png -----------------------------------------------------------
    dh, dw = disp[0].shape
    rgb = np.zeros((dh, dw, 3), np.float32)
    nodata = np.zeros((dh, dw), bool)
    for i, d in enumerate(disp):
        v = d[np.isfinite(d)]
        if v.size == 0:
            err("ratio band %s has no valid pixels" % ORDER[i])
        lo, hi = np.percentile(v, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        rgb[..., i] = np.clip((d - lo) / (hi - lo), 0, 1)
        nodata |= ~np.isfinite(d)
    rgb[nodata] = 0.0                                   # no data -> black

    gt_d = (gt[0], gt[1] * k, 0.0, gt[3], 0.0, gt[5] * k)
    extent = (gt[0], gt[0] + dw * k * gt[1], gt[3] + dh * k * gt[5], gt[3])
    rings = rings_xy(polys)
    lw_px = float(np.clip(dw / 900.0, 1.5, 4.0))
    lds, llyr = mem_layer(dst_srs, [line_geom(xy) for xy in rings])
    line_ds = burn(llyr, dw, dh, gt_d, wkt, all_touched=True)
    under = line_ds.GetRasterBand(1).ReadAsArray() > 0
    line_ds = None
    under = dilate(under, int(np.ceil(lw_px / 2.0)) + 1)
    pix = rgb[under] if under.any() else rgb.reshape(-1, 3)
    colour, cscore = pick_line_colour(pix, rng)
    hexcol = "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in colour)
    print("outline colour %s (contrast score %.1f, from %d pixels along the "
          "outlines)" % (hexcol, cscore, len(pix)))

    rank_txt = "rank %d of %d pairs%s" % (best["rank"], len(results),
                                          " (chosen with --pair)" if a.pair else "")
    title = [
        "T3 ratio post/pre    R = T22   G = T33   B = T11    "
        "(10·log10 ratio, per-band 2–98% stretch)",
        "pre: %s (%s)     post: %s (%s)     Δt = %d days"
        % (pre, dates[pre].isoformat(), post, dates[post].isoformat(), best["days"]),
        "%s: %d polygons on image, outline %s (auto max-contrast)"
        % (shp_name, n_on, hexcol),
        "JM = %.3f (0–2)   polygon coverage = %.0f%%   score = %.3f   %s   "
        "[background: %s]"
        % (best["jm"], 100 * best["coverage"], best["score"], rank_txt, bg_desc),
        "ratio.bin  =  python t3_ratio.py %s %s" % (pre, post),
    ]
    make_png("ratio.png", rgb, extent, rings, colour, lw_px, title)
    print("wrote ratio.png (%d x %d image%s), ratio_scan.csv"
          % (dw, dh, (", %dx%d block mean" % (k, k)) if k > 1 else ""))


if __name__ == "__main__":
    main()
