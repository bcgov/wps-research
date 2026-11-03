#!/usr/bin/env python3
"""raster2kmz.py -- put one or more rasters into a KMZ for Google Earth

Converts GeoTIFF (.tif/.tiff) and ENVI (.bin + .hdr) rasters into a single
KMZ file, either as image overlays (default) or as footprint outlines only.

USAGE
  python3 raster2kmz.py [options] [INPUT ...]

  python3 raster2kmz.py                          # images of every raster here -> rasters.kmz
  python3 raster2kmz.py -f                       # footprints (outlines) only -> footprints.kmz
  python3 raster2kmz.py dir/ -o out.kmz          # images of every raster in a folder
  python3 raster2kmz.py dir/ -f -o out.kmz       # footprints of every raster in a folder
  python3 raster2kmz.py a.tif b.bin              # images of selected files only
  python3 raster2kmz.py stack.bin -b 4,3,2       # RGB from file bands 4,3,2 (S2 true colour)
  python3 raster2kmz.py stack.bin -b 12          # greyscale from the 12th band in the file
  python3 raster2kmz.py *.tif -s minmax -m 8192  # min-max stretch, PNGs up to 8192 px

  INPUT                   raster files and/or directories to convert
                          (default: the current directory)

OPTIONS
  -h, --help              show this help and exit
  -o, --output FILE       output KMZ path (default: rasters.kmz, or
                          footprints.kmz with --footprint)
  -n, --name NAME         name shown for the KMZ in Google Earth
                          (default: output file name without .kmz)

  Footprint mode:
  -f, --footprint         footprint only: draw the outline of each raster's
                          extent instead of its image
  --line-color AABBGGRR   outline colour as 8 hex digits in KML order
                          alpha,blue,green,red (default: ff0000ff = red;
                          ff00ffff = yellow, ffffffff = white)
  --line-width W          outline width in pixels (default: 2)

  Image mode (the default; these are ignored with --footprint):
  -b, --bands LIST        bands to show, 1-based: one band for greyscale
                          (e.g. 1) or three for red,green,blue (e.g. 4,3,2);
                          applied to every input raster
  -s, --stretch MODE      contrast stretch of each band to 0-255:
                            percent  clip at the --percent limits (default)
                            minmax   use the full data range
                            none     just clip values to 0-255
  -p, --percent LOW HIGH  percentiles for --stretch percent (default: 2 98)
  -r, --resample METHOD   resampling when reprojecting: near, bilinear
                          (default), cubic, average or mode; single-band
                          colour-table rasters always use near
  -m, --max-size N        cap on the longest side of each PNG in pixels;
                          larger rasters are downsampled (default: 4096;
                          0 = no cap)

NOTES
  * --footprint draws the outline of each raster's full extent (the
    rectangle of its pixel grid, including any NoData areas) as an
    unfilled polygon, reprojected with densified edges so it stays
    accurate in lat/lon. Clicking an outline in Google Earth shows the
    file name, size, band count, coordinate system and bounds. Pixel
    values are not read, so this is fast even for very large files.
  * ENVI .bin files need a header next to them: name.hdr or name.bin.hdr.
  * Rasters can be in any projection; they are reprojected to WGS84
    lat/lon (EPSG:4326), which is what KML requires.
  * NoData pixels (as declared in the file) and NaN pixels are transparent.
  * Band selection when --bands is not given: a 3- or 4-band raster is
    shown as RGB from bands 1,2,3; anything else is shown as greyscale
    band 1. A single band with a colour table keeps its colours.
  * Requires GDAL's Python bindings (osgeo) and numpy.

CHANGE HISTORY
20260115 original (as rasters_to_kmz.py): convert all .tif and .bin raster
         files in present directory into a KMZ file
20261103 added --help / command-line options and fixed bugs (see FIXES below)
20261103 renamed raster2kmz.py; added -f / --footprint (outline of each
         raster's extent, no imagery); usage moved to the top of the file,
         and -h now prints exactly the text above CHANGE HISTORY

FIXES (20261103)
  1. Shebang was below the module docstring, so ./raster2kmz.py would not
     run directly; it is now on line 1.
  2. Latitude/longitude were swapped with GDAL >= 3: EPSG:4326 uses lat,lon axis
     order unless the traditional GIS order (lon,lat) is requested, so the
     LatLonBox was transposed. The bounds now come straight from the warped
     output, whose geotransform is always lon/lat.
  3. Bounds from only the 4 reprojected corners could be smaller than the true
     footprint (projected edges curve in lon/lat), which clipped the image.
     gdal.Warp now computes the extent itself, so the PNG and the LatLonBox
     match pixel-for-pixel.
  4. scaleParams=[[0, 255]] was not an auto-scale: it mapped source 0..255
     straight to 0..255, so float data (e.g. 0..1 or dB) came out black and
     16-bit data came out saturated. Each band is now contrast-stretched
     (2-98 percentile by default; see --stretch).
  5. NoData / outside-footprint areas were opaque black. They are now
     transparent (PNG alpha band), as are NaN pixels.
  6. Rasters with more than 4 bands (e.g. Sentinel-2 stacks) failed, because
     PNG takes at most 4 bands. Bands are now selected (see --bands).
  7. a.tif and a.bin both wrote files/a.png, so one overlay silently replaced
     the other. PNG names inside the KMZ are now unique.
  8. The .hdr check used str.replace('.bin', '.hdr'), which changed every
     '.bin' in the path and missed upper-case FILE.BIN; it now also accepts
     the file.bin.hdr naming.
  9. Rasters with no georeferencing raised a confusing error; they are now
     skipped with a clear message.
 10. Overlay names were not XML-escaped (a '&' in a filename broke the KML)
     and hrefs were not URL-quoted (spaces).
 11. Files were processed in arbitrary os.listdir order; now sorted.
 12. The script exited with status 0 even when nothing was converted.
 13. Temp VRTs went to a fixed /tmp path; now in-memory (/vsimem/).
 14. Very large rasters produced PNGs too big for Google Earth to load; the
     longest side is now capped (see --max-size).
 15. Single-band rasters with a colour table (classification maps) now keep
     their colours instead of being stretched to greyscale.
"""
import argparse
import os
import sys
import tempfile
import shutil
import zipfile
from urllib.parse import quote
from xml.sax.saxutils import escape

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

RASTER_EXTS = ('.tif', '.tiff', '.bin')

KML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{doc_name}</name>
{overlays}
  </Document>
</kml>
"""

OVERLAY_TEMPLATE = """    <GroundOverlay>
      <name>{name}</name>
      <Icon>
        <href>files/{png_href}</href>
      </Icon>
      <LatLonBox>
        <north>{north:.10f}</north>
        <south>{south:.10f}</south>
        <east>{east:.10f}</east>
        <west>{west:.10f}</west>
      </LatLonBox>
    </GroundOverlay>"""

FOOTPRINT_STYLE = """    <Style id="footprint">
      <LineStyle>
        <color>{color}</color>
        <width>{width}</width>
      </LineStyle>
      <PolyStyle>
        <fill>0</fill>
        <outline>1</outline>
      </PolyStyle>
    </Style>"""

FOOTPRINT_TEMPLATE = """    <Placemark>
      <name>{name}</name>
      <description>{description}</description>
      <styleUrl>#footprint</styleUrl>
      <Polygon>
        <tessellate>1</tessellate>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
{coordinates}
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>"""

# The help text is the top of this file (the module docstring, down to
# CHANGE HISTORY), so the file header and -h output are always identical.
HELP_TEXT = (__doc__ or 'See the top of raster2kmz.py for usage.'
             ).split('CHANGE HISTORY')[0].rstrip() + '\n'


class _HelpAction(argparse.Action):
    def __init__(self, option_strings, dest, **kw):
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS)

    def __call__(self, parser, namespace, values, option_string=None):
        print(HELP_TEXT)
        parser.exit()


def parse_args(argv=None):
    """Command-line options (documented at the top of this file)."""
    p = argparse.ArgumentParser(
        prog='raster2kmz.py', add_help=False,
        usage='python3 raster2kmz.py [options] [INPUT ...]   (-h for help)')
    p.add_argument('-h', '--help', action=_HelpAction)
    p.add_argument('inputs', nargs='*', default=['.'])
    p.add_argument('-o', '--output', default=None)
    p.add_argument('-n', '--name', default=None)
    p.add_argument('-f', '--footprint', action='store_true')
    p.add_argument('--line-color', default='ff0000ff')
    p.add_argument('--line-width', type=float, default=2.0)
    p.add_argument('-b', '--bands', default=None)
    p.add_argument('-s', '--stretch', choices=['percent', 'minmax', 'none'],
                   default='percent')
    p.add_argument('-p', '--percent', nargs=2, type=float, default=[2.0, 98.0])
    p.add_argument('-r', '--resample', default='bilinear',
                   choices=['near', 'bilinear', 'cubic', 'average', 'mode'])
    p.add_argument('-m', '--max-size', type=int, default=4096)
    args = p.parse_args(argv)
    if args.output is None:
        args.output = 'footprints.kmz' if args.footprint else 'rasters.kmz'
    return args


def has_envi_header(path):
    """True if an ENVI .bin has a header as name.hdr or name.bin.hdr."""
    stem = os.path.splitext(path)[0]
    return any(os.path.exists(h) for h in
               (stem + '.hdr', stem + '.HDR', path + '.hdr', path + '.HDR'))


def find_rasters(inputs):
    """Expand files/directories into a sorted list of raster paths."""
    rasters = []
    for item in inputs:
        if os.path.isdir(item):
            candidates = sorted(os.path.join(item, f) for f in os.listdir(item))
        elif os.path.isfile(item):
            candidates = [item]
        else:
            print(f"[WARN] Not found, skipping: {item}")
            continue

        for path in candidates:
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext not in RASTER_EXTS:
                if os.path.isfile(item):  # named explicitly: let GDAL try it
                    rasters.append(path)
                continue
            if ext == '.bin' and not has_envi_header(path):
                print(f"[WARN] Skipping {os.path.basename(path)}: no .hdr file found")
                continue
            rasters.append(path)

    seen, unique = set(), []  # drop duplicates, keep order
    for r in rasters:
        key = os.path.abspath(r)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def wgs84_wkt():
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)  # lon, lat
    return srs.ExportToWkt()


def pick_bands(src, bands_arg):
    """Return the 1-based source bands to display."""
    n = src.RasterCount
    if bands_arg:
        try:
            bands = [int(b) for b in bands_arg.replace(' ', '').split(',') if b]
        except ValueError:
            raise RuntimeError(f"--bands must be integers like 4,3,2, got '{bands_arg}'")
        if len(bands) not in (1, 3):
            raise RuntimeError("--bands needs one band (greyscale) or three (RGB)")
        bad = [b for b in bands if b < 1 or b > n]
        if bad:
            raise RuntimeError(f"band(s) {bad} not in this raster (it has {n} band(s))")
        return bands
    return [1, 2, 3] if n in (3, 4) else [1]


def stretch(arr, valid, mode, pct):
    """Scale one band to uint8 0..255 using only the valid pixels."""
    out = np.zeros(arr.shape, dtype=np.uint8)
    vals = arr[valid]
    if vals.size == 0:
        return out
    if mode == 'none':
        out[valid] = np.clip(vals, 0, 255).astype(np.uint8)
        return out
    if mode == 'minmax':
        lo, hi = float(vals.min()), float(vals.max())
    else:
        lo, hi = (float(v) for v in np.percentile(vals, pct))
    if hi <= lo:  # constant band: show valid pixels as white
        out[valid] = 255
        return out
    scaled = (vals.astype(np.float64) - lo) * (255.0 / (hi - lo))
    out[valid] = np.clip(np.round(scaled), 0, 255).astype(np.uint8)
    return out


def convert_to_png(input_path, output_path, args):
    """Reproject one raster to WGS84 and write it as an RGBA or grey+alpha PNG.
    Returns (west, south, east, north) of the PNG in decimal degrees."""
    src = gdal.Open(input_path)

    if not (src.GetProjection() or src.GetGCPProjection()):
        raise RuntimeError("no coordinate system / georeferencing found; "
                           "cannot place it on the map")

    bands = pick_bands(src, args.bands)
    first = src.GetRasterBand(bands[0])
    ctable = first.GetColorTable() if len(bands) == 1 else None
    resample = 'near' if ctable is not None else args.resample

    tag = f"/vsimem/{abs(hash(input_path))}"
    # Band subset as a VRT (works on all GDAL 3.x; Warp has no band option there)
    sub = gdal.Translate(tag + '_s.vrt', src, format='VRT', bandList=bands)
    warp_kw = dict(format='VRT', dstSRS=wgs84_wkt(), dstAlpha=True,
                   resampleAlg=resample, multithread=True)

    # First pass: let GDAL work out the full lon/lat extent and size
    vrt = gdal.Warp(tag + '_a.vrt', sub, **warp_kw)
    w, h = vrt.RasterXSize, vrt.RasterYSize

    # Second pass at a reduced size if the raster is too big for Google Earth
    if args.max_size and max(w, h) > args.max_size:
        scale = args.max_size / float(max(w, h))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        print(f"  Downsampling {w}x{h} -> {nw}x{nh} (--max-size {args.max_size})")
        gt = vrt.GetGeoTransform()
        bounds = (gt[0], gt[3] + h * gt[5], gt[0] + w * gt[1], gt[3])
        vrt = None
        vrt = gdal.Warp(tag + '_b.vrt', sub, width=nw, height=nh,
                        outputBounds=bounds, **warp_kw)
        w, h = nw, nh

    gt = vrt.GetGeoTransform()
    west, north = gt[0], gt[3]
    east, south = gt[0] + w * gt[1], gt[3] + h * gt[5]

    nb = len(bands)
    alpha = vrt.GetRasterBand(nb + 1).ReadAsArray() > 0
    data = [vrt.GetRasterBand(i + 1).ReadAsArray() for i in range(nb)]
    for d in data:  # NaN counts as missing data
        if np.issubdtype(d.dtype, np.floating):
            alpha &= np.isfinite(d)

    if ctable is not None:  # keep the raster's own colours
        lut = np.zeros((256, 4), dtype=np.uint8)
        for i in range(min(256, ctable.GetCount())):
            lut[i] = ctable.GetColorEntry(i)
        idx = np.clip(data[0], 0, 255).astype(np.uint8)
        rgba = lut[idx]
        rgba[..., 3] = np.where(alpha, rgba[..., 3], 0)
        out_bands = [rgba[..., i] for i in range(4)]
    else:
        out_bands = [stretch(d, alpha, args.stretch, args.percent) for d in data]
        out_bands.append(np.where(alpha, 255, 0).astype(np.uint8))

    mem = gdal.GetDriverByName('MEM').Create('', w, h, len(out_bands), gdal.GDT_Byte)
    for i, arr in enumerate(out_bands):
        mem.GetRasterBand(i + 1).WriteArray(arr)
    mem.GetRasterBand(len(out_bands)).SetColorInterpretation(gdal.GCI_AlphaBand)
    gdal.GetDriverByName('PNG').CreateCopy(output_path, mem)

    mem = vrt = sub = src = None
    for suffix in ('_a.vrt', '_b.vrt', '_s.vrt'):
        try:
            gdal.Unlink(tag + suffix)
        except RuntimeError:
            pass
    return west, south, east, north


def raster_footprint(input_path, points_per_edge=32):
    """Outline of a raster's full extent in lon/lat (WGS84).
    Walks the edge of the pixel grid with points_per_edge points per side, so
    the outline follows any curvature from reprojection. Works for rasters
    georeferenced by geotransform or by GCPs. No pixel values are read.
    Returns (ring, info): ring = closed list of (lon, lat); info = dict."""
    src = gdal.Open(input_path)
    if not (src.GetProjection() or src.GetGCPProjection()):
        raise RuntimeError("no coordinate system / georeferencing found; "
                           "cannot place it on the map")

    w, h = src.RasterXSize, src.RasterYSize
    n = max(2, points_per_edge)
    t = [i / (n - 1) for i in range(n)]
    pix = ([(w * f, 0) for f in t] +             # top, left -> right
           [(w, h * f) for f in t[1:]] +         # right, top -> bottom
           [(w * (1 - f), h) for f in t[1:]] +   # bottom, right -> left
           [(0, h * (1 - f)) for f in t[1:]])    # left, bottom -> top

    tr = gdal.Transformer(src, None, ['DST_SRS=' + wgs84_wkt()])
    pts, ok = tr.TransformPoints(0, pix)
    ring = [(p[0], p[1]) for p, good in zip(pts, ok) if good]
    if len(ring) < 4:
        raise RuntimeError("could not transform the raster edges to lat/lon")
    if ring[0] != ring[-1]:
        ring.append(ring[0])

    srs_wkt = src.GetProjection() or src.GetGCPProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(srs_wkt)
    auth = srs.GetAuthorityName(None)
    code = srs.GetAuthorityCode(None)
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    info = dict(size=f"{w} x {h} pixels",
                bands=src.RasterCount,
                crs=(srs.GetName() or 'unknown') + (f" ({auth}:{code})" if auth and code else ''),
                bounds=(min(lons), min(lats), max(lons), max(lats)))
    src = None
    return ring, info


def create_footprint_kmz(rasters, args):
    """Build a KMZ of footprint outlines only. Returns the number written."""
    output_kmz = args.output
    doc_name = args.name or os.path.splitext(os.path.basename(output_kmz))[0]

    print(f"\nFound {len(rasters)} raster file(s)")
    print(f"Creating footprint-only {output_kmz}...\n")

    placemarks = []
    for raster_path in rasters:
        raster_name = os.path.basename(raster_path)
        print(f"Processing: {raster_name}")
        try:
            ring, info = raster_footprint(raster_path)
            west, south, east, north = info['bounds']
            print(f"  Bounds: W={west:.6f}, S={south:.6f}, "
                  f"E={east:.6f}, N={north:.6f}")
            desc = (f"File: {raster_path}\n"
                    f"Size: {info['size']}\n"
                    f"Bands: {info['bands']}\n"
                    f"Coordinate system: {info['crs']}\n"
                    f"Bounds (W, S, E, N): {west:.6f}, {south:.6f}, "
                    f"{east:.6f}, {north:.6f}")
            coords = "\n".join(f"              {lon:.10f},{lat:.10f},0"
                               for lon, lat in ring)
            placemarks.append(FOOTPRINT_TEMPLATE.format(
                name=escape(os.path.splitext(raster_name)[0]),
                description=escape(desc),
                coordinates=coords))
            print("  OK\n")
        except Exception as e:
            print(f"  Error: {e}\n")

    if not placemarks:
        print("[ERROR] No rasters were successfully processed!")
        return 0

    body = FOOTPRINT_STYLE.format(color=escape(args.line_color),
                                  width=args.line_width)
    kml = KML_TEMPLATE.format(doc_name=escape(doc_name),
                              overlays=body + "\n" + "\n".join(placemarks))

    out_dir = os.path.dirname(os.path.abspath(output_kmz))
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(output_kmz, 'w', zipfile.ZIP_DEFLATED) as kmz:
        kmz.writestr('doc.kml', kml.encode('utf-8'))

    failed = len(rasters) - len(placemarks)
    msg = f"[SUCCESS] Created {output_kmz} with {len(placemarks)} footprint(s)"
    if failed:
        msg += f"; {failed} raster(s) failed (see errors above)"
    print(msg)
    return len(placemarks)


def unique_png_name(raster_path, used):
    """files/<name>.png, made unique when e.g. a.tif and a.bin both exist."""
    base, ext = os.path.splitext(os.path.basename(raster_path))
    name = f"{base}.png"
    if name.lower() in used:
        name = f"{base}_{ext.lstrip('.')}.png"
    n = 2
    while name.lower() in used:
        name = f"{base}_{ext.lstrip('.')}_{n}.png"
        n += 1
    used.add(name.lower())
    return name


def create_kmz(rasters, args):
    """Build the KMZ. Returns the number of overlays written."""
    output_kmz = args.output
    doc_name = args.name or os.path.splitext(os.path.basename(output_kmz))[0]

    print(f"\nFound {len(rasters)} raster file(s)")
    print(f"Creating {output_kmz}...\n")

    temp_dir = tempfile.mkdtemp(prefix='raster2kmz_')
    files_dir = os.path.join(temp_dir, 'files')
    os.makedirs(files_dir)
    overlays, png_files, used = [], [], set()

    try:
        for raster_path in rasters:
            raster_name = os.path.basename(raster_path)
            print(f"Processing: {raster_name}")
            try:
                png_name = unique_png_name(raster_path, used)
                png_path = os.path.join(files_dir, png_name)
                west, south, east, north = convert_to_png(raster_path, png_path, args)
                print(f"  Bounds: W={west:.6f}, S={south:.6f}, "
                      f"E={east:.6f}, N={north:.6f}")
                overlays.append(OVERLAY_TEMPLATE.format(
                    name=escape(os.path.splitext(raster_name)[0]),
                    png_href=escape(quote(png_name)),
                    north=north, south=south, east=east, west=west))
                png_files.append(png_name)
                print("  OK\n")
            except Exception as e:
                used.discard(png_name.lower())
                print(f"  Error: {e}\n")

        if not overlays:
            print("[ERROR] No rasters were successfully processed!")
            return 0

        kml = KML_TEMPLATE.format(doc_name=escape(doc_name),
                                  overlays="\n".join(overlays))

        out_dir = os.path.dirname(os.path.abspath(output_kmz))
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(output_kmz, 'w', zipfile.ZIP_DEFLATED) as kmz:
            kmz.writestr('doc.kml', kml.encode('utf-8'))  # doc.kml goes first
            for png in png_files:
                kmz.write(os.path.join(files_dir, png), f'files/{png}')

        failed = len(rasters) - len(overlays)
        msg = f"[SUCCESS] Created {output_kmz} with {len(overlays)} overlay(s)"
        if failed:
            msg += f"; {failed} raster(s) failed (see errors above)"
        print(msg)
        return len(overlays)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main(argv=None):
    args = parse_args(argv)
    lc = args.line_color.lstrip('#').lower()
    if len(lc) != 8 or any(c not in '0123456789abcdef' for c in lc):
        sys.exit("[ERROR] --line-color must be 8 hex digits in AABBGGRR order, "
                 "e.g. ff0000ff for red")
    args.line_color = lc
    if args.line_width <= 0:
        sys.exit("[ERROR] --line-width must be positive")
    if args.max_size < 0:
        sys.exit("[ERROR] --max-size must be 0 (no cap) or a positive number")
    if not 0 <= args.percent[0] < args.percent[1] <= 100:
        sys.exit("[ERROR] --percent needs 0 <= LOW < HIGH <= 100")

    print("=" * 60)
    print("Raster to KMZ Converter" + (" (footprints only)" if args.footprint else ""))
    print("=" * 60)

    rasters = find_rasters(args.inputs)
    if not rasters:
        print("\n[ERROR] No .tif/.tiff or .bin raster files found in: "
              + ", ".join(args.inputs))
        print("Make sure .bin files have corresponding .hdr files.")
        print("Run with -h for help.")
        sys.exit(1)

    build = create_footprint_kmz if args.footprint else create_kmz
    if build(rasters, args) == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
