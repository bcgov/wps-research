#!/usr/bin/env python3
"""
Convert VIIRS (or generic) netCDF to ENVI Type-4 (Float32) format.

All bands are resampled to the finest resolution present in any subdataset.
Map/CRS info is preserved through multiple fallback strategies:

  Strategy 1: Native GeoTransform + Projection from the netCDF itself
  Strategy 2: CF-convention attributes (grid_mapping, crs_wkt, etc.)
  Strategy 3: GDAL VRT-based geolocation arrays (from companion VNP03IMG)
  Strategy 4: Warp from geolocation arrays to a regular WGS84 grid

Requires: GDAL >= 3.0, numpy, scipy
Optional:  requests (for auto-downloading geolocation files)
"""

from osgeo import gdal, osr, ogr
import numpy as np
import sys
import os
import argparse
import re
import glob
import tempfile
import json

gdal.UseExceptions()


# ─────────────────────────────────────────────────────────────────────────────
# VIIRS filename parsing & geolocation file discovery
# ─────────────────────────────────────────────────────────────────────────────

def parse_viirs_filename(filename):
    """Parse VIIRS filename: VNP02IMG.A2025245.0000.002.2025245091212.nc"""
    basename = os.path.basename(filename)
    parts = basename.split('.')
    if len(parts) < 5:
        return None
    return {
        'product': parts[0],
        'date': parts[1],
        'time': parts[2],
        'collection': parts[3],
        'processing': parts[4].replace('.nc', ''),
    }


def find_local_geolocation_file(img_file):
    """Look for matching VNP03IMG file in the same directory."""
    meta = parse_viirs_filename(img_file)
    if not meta:
        return None
    directory = os.path.dirname(img_file) or '.'
    pattern = f"VNP03IMG.{meta['date']}.{meta['time']}.{meta['collection']}.*.nc"
    matches = glob.glob(os.path.join(directory, pattern))
    if matches:
        print(f"[geo] Found local geolocation file: {matches[0]}")
        return matches[0]
    return None


def download_geolocation_file(img_file):
    """Download matching VNP03IMG from LAADS DAAC (requires EARTHDATA_TOKEN)."""
    try:
        import requests
    except ImportError:
        print("[geo] 'requests' not installed — cannot auto-download.")
        return None

    meta = parse_viirs_filename(img_file)
    if not meta:
        return None

    token = os.environ.get('EARTHDATA_TOKEN')
    if not token:
        print("[geo] EARTHDATA_TOKEN not set. See https://ladsweb.modaps.eosdis.nasa.gov/")
        return None

    year = meta['date'][1:5]
    doy = meta['date'][5:8]
    dir_url = f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5000/VNP03IMG/{year}/{doy}/"
    headers = {'Authorization': f'Bearer {token}'}

    print(f"[geo] Searching LAADS DAAC: {dir_url}")
    try:
        resp = requests.get(dir_url, headers=headers, timeout=30)
        resp.raise_for_status()
        needle = f"VNP03IMG.{meta['date']}.{meta['time']}.{meta['collection']}"
        matches = re.findall(r'href="(' + re.escape(needle) + r'[^"]*\.nc)"', resp.text)
        if not matches:
            print("[geo] No matching file found on server.")
            return None

        geo_url = dir_url + matches[0]
        out_path = os.path.join(os.path.dirname(img_file) or '.', matches[0])
        print(f"[geo] Downloading: {matches[0]}")
        r = requests.get(geo_url, headers=headers, stream=True, timeout=120)
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"[geo] Saved to: {out_path}")
        return out_path
    except Exception as e:
        print(f"[geo] Download failed: {e}")
        return None


def get_geolocation_file(img_file, auto_download=False):
    """Find or download geolocation file."""
    geo = find_local_geolocation_file(img_file)
    if geo:
        return geo
    if auto_download:
        return download_geolocation_file(img_file)
    print("[geo] No local geolocation file found. Use --geo or --auto-download.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# netCDF introspection
# ─────────────────────────────────────────────────────────────────────────────

def get_subdatasets(nc_file):
    """Return list of (name, description) tuples for all subdatasets."""
    ds = gdal.Open(nc_file)
    if ds is None:
        raise ValueError(f"Cannot open {nc_file}")
    sds = ds.GetSubDatasets()
    ds = None
    return sds


def get_dataset_info(sd_name):
    """Return dict with width, height, bands, dtype, geotransform, projection."""
    ds = gdal.Open(sd_name)
    if ds is None:
        return None
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    info = {
        'name': sd_name,
        'width': ds.RasterXSize,
        'height': ds.RasterYSize,
        'bands': ds.RasterCount,
        'dtype': ds.GetRasterBand(1).DataType,
        'geotransform': gt,
        'projection': proj,
        'has_valid_gt': gt is not None and gt != (0, 1, 0, 0, 0, 1),
        'has_projection': proj is not None and proj != '',
        'metadata': ds.GetMetadata(),
    }
    ds = None
    return info


def find_finest_resolution(subdatasets):
    """Return info dict for the subdataset with the most pixels."""
    best = None
    best_px = 0
    for sd_name, sd_desc in subdatasets:
        info = get_dataset_info(sd_name)
        if info is None:
            continue
        px = info['width'] * info['height']
        if px > best_px:
            best_px = px
            best = info
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Native GeoTransform already in the netCDF
# ─────────────────────────────────────────────────────────────────────────────

def try_native_geotransform(subdatasets):
    """
    Check if any subdataset already carries a valid GeoTransform + projection.
    Returns (geotransform, projection_wkt) or (None, None).
    """
    for sd_name, sd_desc in subdatasets:
        info = get_dataset_info(sd_name)
        if info and info['has_valid_gt'] and info['has_projection']:
            print(f"[crs] Native GeoTransform found in: {sd_desc}")
            print(f"       GT = {info['geotransform']}")
            return info['geotransform'], info['projection']
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: CF-convention grid_mapping / crs_wkt
# ─────────────────────────────────────────────────────────────────────────────

def try_cf_convention(nc_file, subdatasets, target_w, target_h):
    """
    Look for CF-convention coordinate variables (x, y, lat, lon) and
    grid_mapping attributes.  Build a GeoTransform from them.
    Returns (geotransform, projection_wkt) or (None, None).
    """
    ds = gdal.Open(nc_file)
    if ds is None:
        return None, None

    # Check root metadata for spatial_ref / crs_wkt
    md = ds.GetMetadata()
    md_all = {}
    for domain in ['', 'GEOLOCATION', 'SUBDATASETS']:
        md_all.update(ds.GetMetadata(domain) or {})

    proj_wkt = None
    for key in ('crs_wkt', 'spatial_ref', 'grid_mapping_name'):
        if key in md_all:
            proj_wkt = md_all[key]
            break

    ds = None

    # Look for 1-D coordinate arrays named x/y or lat/lon
    coord_names = {}
    for sd_name, sd_desc in subdatasets:
        short = sd_name.split(':')[-1].strip().lower()
        if short in ('x', 'y', 'lat', 'lon', 'latitude', 'longitude'):
            coord_names[short] = sd_name

    # Try x/y first, then lon/lat
    x_key = coord_names.get('x') or coord_names.get('lon') or coord_names.get('longitude')
    y_key = coord_names.get('y') or coord_names.get('lat') or coord_names.get('latitude')

    if x_key and y_key:
        xds = gdal.Open(x_key)
        yds = gdal.Open(y_key)
        if xds and yds:
            x_arr = xds.GetRasterBand(1).ReadAsArray().flatten()
            y_arr = yds.GetRasterBand(1).ReadAsArray().flatten()
            xds = None
            yds = None

            if len(x_arr) > 1 and len(y_arr) > 1:
                # Build geotransform from evenly-spaced coordinates
                dx = (x_arr[-1] - x_arr[0]) / (len(x_arr) - 1)
                dy = (y_arr[-1] - y_arr[0]) / (len(y_arr) - 1)
                # pixel-edge convention
                gt = (
                    float(x_arr[0] - dx / 2),
                    float(dx),
                    0.0,
                    float(y_arr[0] - dy / 2),
                    0.0,
                    float(dy),
                )
                if proj_wkt is None:
                    # Assume WGS84 geographic if coordinates look like degrees
                    if abs(x_arr[0]) <= 360 and abs(y_arr[0]) <= 90:
                        srs = osr.SpatialReference()
                        srs.ImportFromEPSG(4326)
                        proj_wkt = srs.ExportToWkt()

                if proj_wkt:
                    print(f"[crs] CF-convention coordinates found")
                    print(f"       GT = {gt}")
                    return gt, proj_wkt

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3 & 4: Geolocation arrays from companion file
# ─────────────────────────────────────────────────────────────────────────────

def read_geolocation_arrays(geo_file):
    """Read lat/lon arrays from a geolocation netCDF (e.g. VNP03IMG)."""
    sds = get_subdatasets(geo_file)
    lat = lon = None
    lat_name = lon_name = None

    for sd_name, sd_desc in sds:
        short = sd_name.split(':')[-1].strip().lower()
        desc_low = sd_desc.lower()
        if short == 'latitude' or (lat is None and 'latitude' in desc_low and 'longitude' not in desc_low):
            ds = gdal.Open(sd_name)
            if ds:
                lat = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
                lat_name = sd_name
                ds = None
        elif short == 'longitude' or (lon is None and 'longitude' in desc_low and 'latitude' not in desc_low):
            ds = gdal.Open(sd_name)
            if ds:
                lon = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
                lon_name = sd_name
                ds = None

    if lat is not None and lon is not None:
        print(f"[geo] Lat array shape: {lat.shape}, Lon array shape: {lon.shape}")
        return lat, lon, lat_name, lon_name
    return None, None, None, None


def try_geoloc_vrt_warp(nc_file, geo_file, all_bands, band_names,
                         target_h, target_w, target_res=None):
    """
    Strategy 3/4: Use GDAL's geolocation-array warping to produce a
    properly georeferenced output.

    This creates a VRT with GEOLOCATION metadata pointing at the lat/lon
    arrays, then uses gdal.Warp to resample onto a regular WGS84 grid.

    Returns (warped_bands, geotransform, projection_wkt) or (None, None, None).
    """
    lat, lon, lat_sd, lon_sd = read_geolocation_arrays(geo_file)
    if lat is None:
        return None, None, None

    from scipy.ndimage import zoom

    # Resample lat/lon to match data dimensions if needed
    if lat.shape != (target_h, target_w):
        print(f"[geo] Resampling lat/lon from {lat.shape} → ({target_h}, {target_w})")
        zy = target_h / lat.shape[0]
        zx = target_w / lat.shape[1]
        lat = zoom(lat, (zy, zx), order=1)
        lon = zoom(lon, (zy, zx), order=1)

    # Compute bounds from geolocation arrays (ignoring fill values)
    valid = (np.abs(lat) <= 90) & (np.abs(lon) <= 360)
    if not np.any(valid):
        print("[geo] No valid lat/lon values found!")
        return None, None, None

    min_lon = float(np.nanmin(lon[valid]))
    max_lon = float(np.nanmax(lon[valid]))
    min_lat = float(np.nanmin(lat[valid]))
    max_lat = float(np.nanmax(lat[valid]))
    print(f"[geo] Extent: lon [{min_lon:.4f}, {max_lon:.4f}], lat [{min_lat:.4f}, {max_lat:.4f}]")

    # Determine output pixel size
    if target_res:
        res_x = res_y = target_res
    else:
        # Estimate from median spacing in the geolocation arrays
        if lon.shape[1] > 1:
            dx_samples = np.abs(np.diff(lon[:, ::max(1, lon.shape[1]//100)], axis=1))
            dx_samples = dx_samples[dx_samples > 0]
            res_x = float(np.median(dx_samples)) if len(dx_samples) > 0 else (max_lon - min_lon) / target_w
        else:
            res_x = (max_lon - min_lon) / target_w
        if lat.shape[0] > 1:
            dy_samples = np.abs(np.diff(lat[::max(1, lat.shape[0]//100), :], axis=0))
            dy_samples = dy_samples[dy_samples > 0]
            res_y = float(np.median(dy_samples)) if len(dy_samples) > 0 else (max_lat - min_lat) / target_h
        else:
            res_y = (max_lat - min_lat) / target_h

    print(f"[geo] Estimated pixel size: {res_x:.6f}° × {res_y:.6f}°")

    # ── Write temporary lat/lon rasters so VRT GEOLOCATION can reference them ──
    tmpdir = tempfile.mkdtemp(prefix='viirs_geo_')
    lat_tif = os.path.join(tmpdir, 'lat.tif')
    lon_tif = os.path.join(tmpdir, 'lon.tif')
    _write_array_to_tif(lat, lat_tif)
    _write_array_to_tif(lon, lon_tif)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    proj_wkt = srs.ExportToWkt()

    warped_bands = []

    for idx, (data, name) in enumerate(zip(all_bands, band_names)):
        # Write this band to a temp GeoTIFF
        src_tif = os.path.join(tmpdir, f'band_{idx}.tif')
        _write_array_to_tif(data, src_tif)

        # Build VRT with GEOLOCATION domain
        vrt_path = os.path.join(tmpdir, f'band_{idx}.vrt')
        _write_geoloc_vrt(src_tif, lon_tif, lat_tif, target_w, target_h, vrt_path)

        # Warp to regular grid
        dst_tif = os.path.join(tmpdir, f'band_{idx}_warped.tif')
        try:
            gdal.Warp(
                dst_tif, vrt_path,
                format='GTiff',
                dstSRS='EPSG:4326',
                outputBounds=(min_lon, min_lat, max_lon, max_lat),
                xRes=res_x,
                yRes=res_y,
                resampleAlg='bilinear',
                geoloc=True,
                errorThreshold=0.125,
                dstNodata=np.nan,
            )
            wds = gdal.Open(dst_tif)
            if wds:
                warped_bands.append(wds.GetRasterBand(1).ReadAsArray().astype(np.float32))
                if idx == 0:
                    gt = wds.GetGeoTransform()
                    out_w = wds.RasterXSize
                    out_h = wds.RasterYSize
                wds = None
            else:
                print(f"[warp] Failed to open warped output for band {name}")
                warped_bands.append(data.astype(np.float32))
        except Exception as e:
            print(f"[warp] Error warping band {name}: {e}")
            warped_bands.append(data.astype(np.float32))

    # Clean up temp files
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    if warped_bands and gt:
        print(f"[crs] Warped output: {out_w}×{out_h}, GT={gt}")
        return warped_bands, gt, proj_wkt
    return None, None, None


def _write_array_to_tif(arr, path):
    """Write a 2D numpy array to a single-band GeoTIFF (no projection)."""
    h, w = arr.shape
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(path, w, h, 1, gdal.GDT_Float64)
    # Simple pixel-is-area transform (identity)
    ds.SetGeoTransform((0, 1, 0, 0, 0, 1))
    ds.GetRasterBand(1).WriteArray(arr)
    ds.GetRasterBand(1).SetNoDataValue(np.nan)
    ds.FlushCache()
    ds = None


def _write_geoloc_vrt(data_tif, lon_tif, lat_tif, width, height, vrt_path):
    """Write a VRT file that attaches GEOLOCATION arrays to a data raster."""
    vrt_xml = f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
  <Metadata domain="GEOLOCATION">
    <MDI key="X_DATASET">{lon_tif}</MDI>
    <MDI key="X_BAND">1</MDI>
    <MDI key="Y_DATASET">{lat_tif}</MDI>
    <MDI key="Y_BAND">1</MDI>
    <MDI key="PIXEL_OFFSET">0</MDI>
    <MDI key="LINE_OFFSET">0</MDI>
    <MDI key="PIXEL_STEP">1</MDI>
    <MDI key="LINE_STEP">1</MDI>
    <MDI key="SRS">GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]</MDI>
  </Metadata>
  <VRTRasterBand dataType="Float64" band="1">
    <NoDataValue>nan</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{data_tif}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />
      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    with open(vrt_path, 'w') as f:
        f.write(vrt_xml)


# ─────────────────────────────────────────────────────────────────────────────
# Band resampling
# ─────────────────────────────────────────────────────────────────────────────

def resample_array(data, target_h, target_w):
    """Bilinear resample to target dimensions."""
    if data.shape == (target_h, target_w):
        return data
    from scipy.ndimage import zoom
    factors = (target_h / data.shape[0], target_w / data.shape[1])
    return zoom(data, factors, order=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main conversion
# ─────────────────────────────────────────────────────────────────────────────

def netcdf_to_envi(nc_file, output_file, geo_file=None, bands_only=False,
                    pattern=None, target_res=None, no_warp=False):
    """
    Convert netCDF to georeferenced ENVI Type-4 (Float32).

    Parameters
    ----------
    nc_file : str       Input netCDF
    output_file : str   Output .bin (ENVI)
    geo_file : str      Companion geolocation file (VNP03IMG) or None
    bands_only : bool   Skip quality_flags / uncert_index
    pattern : str       Regex filter for subdataset names
    target_res : float  Force output pixel size in degrees
    no_warp : bool      Skip warping; use simple corner-based geotransform
    """
    print(f"{'='*70}")
    print(f"Input : {nc_file}")
    print(f"Output: {output_file}")
    print(f"{'='*70}")

    # ── 1. Enumerate subdatasets ──────────────────────────────────────────
    subdatasets = get_subdatasets(nc_file)
    if not subdatasets:
        # Single-raster netCDF (no subdatasets)
        subdatasets = [(nc_file, os.path.basename(nc_file))]

    print(f"\n[info] Found {len(subdatasets)} subdatasets:")
    for i, (n, d) in enumerate(subdatasets, 1):
        info = get_dataset_info(n)
        sz = f"{info['width']}×{info['height']}" if info else "?"
        print(f"  {i:3d}. [{sz:>12s}] {d}")

    # ── 2. Find finest resolution ─────────────────────────────────────────
    ref = find_finest_resolution(subdatasets)
    if ref is None:
        print("[error] Cannot determine reference dimensions.")
        return False

    target_w, target_h = ref['width'], ref['height']
    print(f"\n[info] Target dimensions (finest): {target_w} × {target_h}")

    # ── 3. Strategy 1: Check for native GeoTransform ─────────────────────
    geotransform, projection = try_native_geotransform(subdatasets)

    # ── 4. Strategy 2: CF-convention coordinates ──────────────────────────
    if geotransform is None:
        geotransform, projection = try_cf_convention(nc_file, subdatasets, target_w, target_h)

    # Decide whether we need geolocation-array warping
    need_warp = (geotransform is None) and (not no_warp) and (geo_file is not None)

    # ── 5. Read all requested bands ───────────────────────────────────────
    print(f"\n[info] Reading bands...")
    all_bands = []
    band_names = []

    try:
        from scipy.ndimage import zoom as _zoom
        has_scipy = True
    except ImportError:
        has_scipy = False
        print("[warn] scipy not installed — bands with mismatched sizes will be skipped")

    for sd_name, sd_desc in subdatasets:
        # Filters
        if bands_only and ('quality_flags' in sd_desc.lower() or 'uncert_index' in sd_desc.lower()):
            continue
        if pattern and not re.search(pattern, sd_desc, re.IGNORECASE):
            continue

        ds = gdal.Open(sd_name)
        if ds is None:
            continue

        w, h = ds.RasterXSize, ds.RasterYSize
        var_name = sd_name.split(':')[-1].replace('/', '_')

        for bi in range(1, ds.RasterCount + 1):
            data = ds.GetRasterBand(bi).ReadAsArray()
            if data is None:
                continue

            # Handle fill / scale / offset from band metadata
            md = ds.GetRasterBand(bi).GetMetadata()
            fill = md.get('_FillValue')
            scale = md.get('scale_factor')
            offset = md.get('add_offset')

            data = data.astype(np.float64)
            if fill is not None:
                data[data == float(fill)] = np.nan
            if scale is not None:
                data *= float(scale)
            if offset is not None:
                data += float(offset)

            # Resample to target dimensions
            if (h, w) != (target_h, target_w):
                if has_scipy:
                    data = resample_array(data, target_h, target_w)
                else:
                    continue

            bname = f"{var_name}_b{bi}" if ds.RasterCount > 1 else var_name
            all_bands.append(data.astype(np.float32))
            band_names.append(bname)
            print(f"  + {bname}  ({w}×{h} → {target_w}×{target_h})")

        ds = None

    print(f"\n[info] Collected {len(all_bands)} bands")
    if not all_bands:
        print("[error] No bands to write!")
        return False

    # ── 6. Strategy 3/4: Geolocation-array warp ──────────────────────────
    if need_warp:
        print(f"\n[crs] No native georeferencing. Warping via geolocation arrays...")
        warped, gt_w, proj_w = try_geoloc_vrt_warp(
            nc_file, geo_file, all_bands, band_names,
            target_h, target_w, target_res
        )
        if warped is not None:
            all_bands = warped
            geotransform = gt_w
            projection = proj_w
            # Dimensions may have changed after warp
            target_h, target_w = all_bands[0].shape

    # ── 7. Fallback: simple corner-based geotransform ─────────────────────
    if geotransform is None and geo_file is not None:
        print(f"\n[crs] Falling back to corner-based approximate geotransform")
        lat, lon, _, _ = read_geolocation_arrays(geo_file)
        if lat is not None:
            if lat.shape != (target_h, target_w):
                lat = resample_array(lat, target_h, target_w)
                lon = resample_array(lon, target_h, target_w)
            valid = (np.abs(lat) <= 90) & (np.abs(lon) <= 360)
            if np.any(valid):
                min_lon = float(np.nanmin(lon[valid]))
                max_lon = float(np.nanmax(lon[valid]))
                min_lat = float(np.nanmin(lat[valid]))
                max_lat = float(np.nanmax(lat[valid]))
                px = (max_lon - min_lon) / target_w
                py = -(max_lat - min_lat) / target_h
                geotransform = (min_lon, px, 0.0, max_lat, 0.0, py)
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(4326)
                projection = srs.ExportToWkt()
                print(f"       GT = {geotransform}")

    # ── 8. Write ENVI file ────────────────────────────────────────────────
    print(f"\n[write] Creating ENVI file: {output_file}")
    driver = gdal.GetDriverByName('ENVI')
    out_ds = driver.Create(output_file, target_w, target_h, len(all_bands), gdal.GDT_Float32)

    if geotransform is not None:
        out_ds.SetGeoTransform(geotransform)
        print(f"  GeoTransform: {geotransform}")
    else:
        print("  [warn] No geotransform available — output will lack georeferencing")

    if projection:
        out_ds.SetProjection(projection)
        srs = osr.SpatialReference(wkt=projection)
        auth = srs.GetAuthorityCode(None)
        print(f"  Projection: {srs.GetName()} (EPSG:{auth})" if auth else f"  Projection: {srs.GetName()}")
    else:
        print("  [warn] No projection available")

    for idx, (data, name) in enumerate(zip(all_bands, band_names), 1):
        band = out_ds.GetRasterBand(idx)
        band.WriteArray(data)
        band.SetDescription(name)
        band.SetNoDataValue(np.nan)
        band.FlushCache()
        print(f"  Band {idx:3d}/{len(all_bands)}: {name}")

    out_ds.FlushCache()
    out_ds = None

    # ── 9. Verify output ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Verifying output...")
    vds = gdal.Open(output_file)
    if vds:
        gt = vds.GetGeoTransform()
        pr = vds.GetProjection()
        print(f"  Size:       {vds.RasterXSize} × {vds.RasterYSize}")
        print(f"  Bands:      {vds.RasterCount}")
        print(f"  GeoTransform valid: {gt is not None and gt != (0, 1, 0, 0, 0, 1)}")
        print(f"  Projection valid:   {pr is not None and pr != ''}")
        if gt and gt != (0, 1, 0, 0, 0, 1):
            ulx = gt[0]
            uly = gt[3]
            lrx = gt[0] + gt[1] * vds.RasterXSize
            lry = gt[3] + gt[5] * vds.RasterYSize
            print(f"  UL corner:  ({ulx:.6f}, {uly:.6f})")
            print(f"  LR corner:  ({lrx:.6f}, {lry:.6f})")
            print(f"  Pixel size: {abs(gt[1]):.6f}° × {abs(gt[5]):.6f}°")
        vds = None

    hdr = output_file.rsplit('.', 1)[0] + '.hdr'
    print(f"\n  Output:  {output_file}")
    print(f"  Header:  {hdr}")
    print(f"{'='*70}")

    # ── 10. Print .hdr contents for inspection ────────────────────────────
    if os.path.exists(hdr):
        print(f"\n[hdr] Contents of {os.path.basename(hdr)}:")
        with open(hdr) as f:
            for line in f:
                print(f"  {line.rstrip()}")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Convert netCDF to georeferenced ENVI Type-4 (Float32)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-find local VNP03IMG geolocation file, warp to WGS84 grid
  python3 netcdf_to_envi.py VNP02IMG.A2025245.0000.002.*.nc output.bin

  # Specify geolocation file explicitly
  python3 netcdf_to_envi.py VNP02IMG.nc output.bin --geo VNP03IMG.nc

  # Only imagery bands, force 0.005° resolution
  python3 netcdf_to_envi.py VNP02IMG.nc output.bin --bands-only --resolution 0.005

  # Only I-bands 1-3
  python3 netcdf_to_envi.py VNP02IMG.nc output.bin --pattern "I0[1-3]"

  # Skip warping (use approximate corner-based geotransform)
  python3 netcdf_to_envi.py VNP02IMG.nc output.bin --geo VNP03IMG.nc --no-warp

  # Auto-download geolocation from LAADS DAAC
  EARTHDATA_TOKEN=xxx python3 netcdf_to_envi.py VNP02IMG.nc output.bin --auto-download
""")
    p.add_argument('input', help='Input netCDF file')
    p.add_argument('output', help='Output ENVI .bin file')
    p.add_argument('--geo', help='Geolocation file (VNP03IMG*.nc)')
    p.add_argument('--bands-only', action='store_true',
                   help='Exclude quality_flags and uncert_index')
    p.add_argument('--pattern', help='Regex to filter subdataset names')
    p.add_argument('--resolution', type=float,
                   help='Force output pixel size in degrees')
    p.add_argument('--no-warp', action='store_true',
                   help='Skip geolocation warping; use corner-based approx.')
    p.add_argument('--auto-download', action='store_true',
                   help='Auto-download geolocation from LAADS DAAC')

    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"[error] File not found: {args.input}")
        sys.exit(1)

    # Resolve geolocation file
    geo = args.geo
    if geo and not os.path.exists(geo):
        print(f"[error] Geo file not found: {geo}")
        sys.exit(1)
    if not geo:
        geo = get_geolocation_file(args.input, auto_download=args.auto_download)

    # Check scipy
    try:
        import scipy
    except ImportError:
        print("[error] scipy is required: pip install scipy")
        sys.exit(1)

    ok = netcdf_to_envi(
        args.input, args.output,
        geo_file=geo,
        bands_only=args.bands_only,
        pattern=args.pattern,
        target_res=args.resolution,
        no_warp=args.no_warp,
    )
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

