#!/usr/bin/env python3
"""
viirs_download_convert.py — Download VIIRS data from LAADS DAAC and convert to
georeferenced ENVI Type-4 (Float32).

Combines:
  1. LAADS DAAC V2 API download (with netCDF integrity checking)
  2. netCDF → ENVI conversion with proper georeferencing via VNP03IMG

Workflow:
  - Downloads VNP02IMG (radiance) + VNP03IMG (geolocation) for each granule
  - Converts each VNP02IMG to ENVI using the matching VNP03IMG for georeferencing
  - Warps swath data onto a regular WGS84 grid via GDAL geolocation arrays

Requires: GDAL >= 3.0, numpy, scipy, netCDF4
"""

from osgeo import gdal, osr
import numpy as np
import sys
import os
import argparse
import re
import glob
import json
import ssl
import shutil
import tempfile
import datetime
from multiprocessing import Pool

# Suppress noisy GDAL warnings for swath netCDF
gdal.UseExceptions()
gdal.SetConfigOption('CPL_LOG', '/dev/null')
os.environ['CPL_LOG'] = '/dev/null'

USERAGENT = 'viirs_download_convert/2.0--' + sys.version.replace('\n', '').replace('\r', '')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: LAADS DAAC DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def geturl(url, token=None, out=None):
    """Fetch URL content. Falls back to cURL if TLS fails."""
    headers = {'user-agent': USERAGENT}
    if token:
        headers['Authorization'] = 'Bearer ' + token

    try:
        CTX = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        from urllib.request import urlopen, Request
        from urllib.error import URLError, HTTPError
        try:
            fh = urlopen(Request(url, headers=headers), context=CTX)
            if out is None:
                return fh.read().decode('utf-8')
            else:
                shutil.copyfileobj(fh, out)
        except HTTPError as e:
            print(f'  HTTP error {e.code}', file=sys.stderr)
            return _getcurl(url, headers, out)
        except URLError as e:
            print(f'  URL error: {e.reason}', file=sys.stderr)
            return _getcurl(url, headers, out)
        return None
    except AttributeError:
        return _getcurl(url, headers, out)


def _getcurl(url, headers, out=None):
    """Fallback to cURL for systems with broken TLS."""
    import subprocess
    try:
        args = ['curl', '--fail', '-sS', '-L', '-b', 'session', '--get', url]
        for k, v in headers.items():
            args.extend(['-H', f'{k}: {v}'])
        if out is None:
            result = subprocess.check_output(args)
            return result.decode('utf-8') if isinstance(result, bytes) else result
        else:
            subprocess.call(args, stdout=out)
    except subprocess.CalledProcessError as e:
        print(f'  cURL error: {e.output}', file=sys.stderr)
    return None


def verify_netcdf(path):
    """Verify a netCDF file can be opened. Returns True if valid."""
    try:
        import netCDF4 as nC
        ds = nC.Dataset(path, 'r')
        ds.close()
        return True
    except Exception:
        return False


def sync_download(src_url, dest_dir, token):
    """
    Download all files from a LAADS DAAC V2 API URL to dest_dir.
    Skips existing valid files. Re-downloads corrupt ones.
    """
    response = geturl(src_url + '.json', token)
    if not response:
        print(f'[download] Failed to fetch listing from {src_url}')
        return []

    files = json.loads(response)
    downloaded = []

    for f in files.get('content', []):
        filesize = int(f['size'])
        path = os.path.join(dest_dir, f['name'])
        url = f['downloadsLink']

        if filesize == 0:
            # Directory entry
            os.makedirs(path, exist_ok=True)
            downloaded.extend(sync_download(src_url + '/' + f['name'], path, token))
            continue

        # Check existing file integrity
        if os.path.exists(path) and os.path.getsize(path) > 0:
            if path.endswith('.nc') and not verify_netcdf(path):
                print(f'[download] Corrupt file, re-downloading: {f["name"]}')
                os.remove(path)
            else:
                print(f'[download] Valid, skipping: {f["name"]}')
                downloaded.append(path)
                continue

        # Download
        print(f'[download] Downloading: {f["name"]}')
        try:
            with open(path, 'w+b') as fh:
                geturl(url, token, fh)
            downloaded.append(path)
        except IOError as e:
            print(f'[download] Error writing {path}: {e}', file=sys.stderr)

    return downloaded


def build_laads_url(product, year, doy, regions=None):
    """Build LAADS DAAC V2 API URL for a given product/date/region."""
    url = (f"https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details?"
           f"products={product}&temporalRanges={year}-{doy}")
    if regions:
        url += f"&regions=%5BAA%5D{regions}"
    return url


def download_day(product, dt, base_dir, token, regions=None):
    """
    Download a single day of VIIRS data for a given product.
    Returns list of downloaded file paths.
    """
    year = dt.year
    doy = dt.timetuple().tm_yday
    dest = os.path.join(base_dir, product, f"{year:04d}", f"{doy:03d}")
    os.makedirs(dest, exist_ok=True)

    url = build_laads_url(product, year, doy, regions)
    print(f'\n[download] {product} {year}/{doy:03d} → {dest}')
    print(f'[download] URL: {url}')

    return sync_download(url, dest, token)


def download_viirs_pair(dt, base_dir, token, regions=None,
                         img_product='VNP02IMG', geo_product='VNP03IMG'):
    """
    Download both radiance (VNP02IMG) and geolocation (VNP03IMG) for a day.
    Returns list of (img_path, geo_path) tuples matched by time.
    """
    img_files = download_day(img_product, dt, base_dir, token, regions)
    geo_files = download_day(geo_product, dt, base_dir, token, regions)

    # Match files by date.time pattern
    pairs = []
    geo_by_key = {}
    for gf in geo_files:
        m = parse_viirs_filename(gf)
        if m:
            geo_by_key[f"{m['date']}.{m['time']}"] = gf

    for imf in img_files:
        m = parse_viirs_filename(imf)
        if m:
            key = f"{m['date']}.{m['time']}"
            gf = geo_by_key.get(key)
            if gf:
                pairs.append((imf, gf))
            else:
                print(f'[match] No geo file for {os.path.basename(imf)}')

    print(f'[match] {len(pairs)} image/geo pairs for {dt.strftime("%Y-%m-%d")}')
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: VIIRS FILENAME PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_viirs_filename(filename):
    """Parse VIIRS filename: VNP02IMG.A2025245.0000.002.2025245091212.nc"""
    parts = os.path.basename(filename).split('.')
    if len(parts) < 5:
        return None
    return {
        'product': parts[0], 'date': parts[1], 'time': parts[2],
        'collection': parts[3],
    }


def find_local_geo_file(img_file):
    """Find matching VNP03IMG in same directory."""
    m = parse_viirs_filename(img_file)
    if not m:
        return None
    d = os.path.dirname(img_file) or '.'
    pat = f"VNP03IMG.{m['date']}.{m['time']}.{m['collection']}.*.nc"
    hits = glob.glob(os.path.join(d, pat))
    if hits:
        return hits[0]
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: NETCDF → ENVI CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

def get_subdatasets(nc_file):
    ds = gdal.Open(nc_file)
    if ds is None:
        raise ValueError(f"Cannot open {nc_file}")
    sds = ds.GetSubDatasets()
    ds = None
    return sds


def sd_info(sd_name):
    ds = gdal.Open(sd_name)
    if not ds:
        return None
    gt = ds.GetGeoTransform()
    pr = ds.GetProjection()
    i = {
        'w': ds.RasterXSize, 'h': ds.RasterYSize,
        'bands': ds.RasterCount,
        'gt': gt, 'proj': pr,
        'has_gt': gt is not None and gt != (0, 1, 0, 0, 0, 1),
        'has_proj': pr is not None and pr != '',
    }
    ds = None
    return i


def finest_dims(subdatasets):
    best_w = best_h = best_px = 0
    for n, _ in subdatasets:
        info = sd_info(n)
        if not info:
            continue
        px = info['w'] * info['h']
        if px > best_px:
            best_px = px
            best_w, best_h = info['w'], info['h']
    return best_w, best_h


def try_native_gt(subdatasets):
    """Strategy 1: check if any subdataset has a real GeoTransform."""
    for n, d in subdatasets:
        info = sd_info(n)
        if info and info['has_gt'] and info['has_proj']:
            print(f'[crs:1] Native GeoTransform in: {d}')
            return info['gt'], info['proj']
    return None, None


def try_cf_coords(subdatasets):
    """Strategy 2: CF-convention 1-D coordinate arrays."""
    coord_sds = {}
    for n, d in subdatasets:
        short = n.split(':')[-1].strip().lower()
        if short in ('x', 'y', 'lat', 'lon', 'latitude', 'longitude'):
            coord_sds[short] = n

    xn = coord_sds.get('x') or coord_sds.get('lon') or coord_sds.get('longitude')
    yn = coord_sds.get('y') or coord_sds.get('lat') or coord_sds.get('latitude')
    if not (xn and yn):
        return None, None

    xds, yds = gdal.Open(xn), gdal.Open(yn)
    if not (xds and yds):
        return None, None
    xa = xds.GetRasterBand(1).ReadAsArray().flatten()
    ya = yds.GetRasterBand(1).ReadAsArray().flatten()
    xds = yds = None

    if len(xa) < 2 or len(ya) < 2:
        return None, None

    dx = (xa[-1] - xa[0]) / (len(xa) - 1)
    dy = (ya[-1] - ya[0]) / (len(ya) - 1)
    gt = (float(xa[0] - dx / 2), float(dx), 0.0,
          float(ya[0] - dy / 2), 0.0, float(dy))

    if abs(xa[0]) <= 360 and abs(ya[0]) <= 90:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        print('[crs:2] CF-convention coordinate arrays found')
        return gt, srs.ExportToWkt()
    return None, None


def read_latlon(geo_file):
    """Read lat/lon arrays from VNP03IMG."""
    sds = get_subdatasets(geo_file)
    lat = lon = None
    for n, d in sds:
        short = n.split(':')[-1].strip().lower()
        dl = d.lower()
        if short == 'latitude' or ('latitude' in dl and 'longitude' not in dl and lat is None):
            ds = gdal.Open(n)
            if ds:
                lat = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
                ds = None
        elif short == 'longitude' or ('longitude' in dl and 'latitude' not in dl and lon is None):
            ds = gdal.Open(n)
            if ds:
                lon = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
                ds = None
    if lat is not None and lon is not None:
        print(f'[geo] Lat/lon shape: {lat.shape}')
        return lat, lon
    print('[geo] Could not find lat/lon arrays!')
    return None, None


def resample_2d(arr, th, tw):
    if arr.shape == (th, tw):
        return arr
    from scipy.ndimage import zoom
    return zoom(arr, (th / arr.shape[0], tw / arr.shape[1]), order=1)


def _write_tif(arr, path):
    h, w = arr.shape
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(path, w, h, 1, gdal.GDT_Float64)
    ds.SetGeoTransform((0, 1, 0, 0, 0, 1))
    ds.GetRasterBand(1).WriteArray(arr)
    ds.FlushCache()
    ds = None


def warp_bands_via_geolocation(all_bands, lat, lon, target_h, target_w,
                                target_res=None):
    """
    Strategy 3: Warp all bands from swath → regular WGS84 grid using
    GDAL's geolocation-array-aware warper. All bands warped in one pass.
    """
    valid = (np.abs(lat) <= 90) & (np.abs(lon) <= 360) & np.isfinite(lat) & np.isfinite(lon)
    if not np.any(valid):
        print('[warp] No valid lat/lon!')
        return None, None, None

    min_lon, max_lon = float(np.min(lon[valid])), float(np.max(lon[valid]))
    min_lat, max_lat = float(np.min(lat[valid])), float(np.max(lat[valid]))
    print(f'[warp] Bounds: lon [{min_lon:.4f}, {max_lon:.4f}]  lat [{min_lat:.4f}, {max_lat:.4f}]')

    # Antimeridian crossing
    if max_lon - min_lon > 350:
        print('[warp] Antimeridian crossing — adjusting')
        lon = lon.copy()
        lon[lon < 0] += 360
        v2 = np.isfinite(lon) & (np.abs(lat) <= 90)
        min_lon, max_lon = float(np.min(lon[v2])), float(np.max(lon[v2]))

    # Estimate pixel size
    if target_res:
        res = target_res
    else:
        sx = max(1, lon.shape[1] // 100)
        sy = max(1, lon.shape[0] // 100)
        dx = np.abs(np.diff(lon[:, ::sx], axis=1))
        dy = np.abs(np.diff(lat[::sy, :], axis=0))
        dx, dy = dx[(dx > 0) & (dx < 1)], dy[(dy > 0) & (dy < 1)]
        rx = float(np.median(dx)) if len(dx) else (max_lon - min_lon) / target_w
        ry = float(np.median(dy)) if len(dy) else (max_lat - min_lat) / target_h
        res = min(rx, ry)
    print(f'[warp] Pixel size: {res:.6f}°')

    tmpdir = tempfile.mkdtemp(prefix='viirs_warp_')
    try:
        nbands = len(all_bands)
        h, w = target_h, target_w

        _write_tif(lat, os.path.join(tmpdir, 'lat.tif'))
        _write_tif(lon, os.path.join(tmpdir, 'lon.tif'))

        # Stack all bands into one multi-band GeoTIFF
        data_tif = os.path.join(tmpdir, 'data.tif')
        drv = gdal.GetDriverByName('GTiff')
        ds = drv.Create(data_tif, w, h, nbands, gdal.GDT_Float32)
        ds.SetGeoTransform((0, 1, 0, 0, 0, 1))
        for i, arr in enumerate(all_bands):
            ds.GetRasterBand(i + 1).WriteArray(arr)
            ds.GetRasterBand(i + 1).SetNoDataValue(-9999.0)
        ds.FlushCache()
        ds = None

        # Build VRT with GEOLOCATION metadata
        wgs84_wkt = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",'
                     '6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
                     'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,'
                     'AUTHORITY["EPSG","8901"]],UNIT["degree",'
                     '0.0174532925199433,AUTHORITY["EPSG","9122"]],'
                     'AUTHORITY["EPSG","4326"]]')

        lat_tif = os.path.join(tmpdir, 'lat.tif')
        lon_tif = os.path.join(tmpdir, 'lon.tif')

        band_xml = '\n'.join([
            f'  <VRTRasterBand dataType="Float32" band="{i+1}">\n'
            f'    <NoDataValue>-9999</NoDataValue>\n'
            f'    <SimpleSource>\n'
            f'      <SourceFilename relativeToVRT="0">{data_tif}</SourceFilename>\n'
            f'      <SourceBand>{i+1}</SourceBand>\n'
            f'      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}" />\n'
            f'      <DstRect xOff="0" yOff="0" xSize="{w}" ySize="{h}" />\n'
            f'    </SimpleSource>\n'
            f'  </VRTRasterBand>'
            for i in range(nbands)
        ])

        vrt_xml = (
            f'<VRTDataset rasterXSize="{w}" rasterYSize="{h}">\n'
            f'  <SRS>{wgs84_wkt}</SRS>\n'
            f'  <Metadata domain="GEOLOCATION">\n'
            f'    <MDI key="X_DATASET">{lon_tif}</MDI>\n'
            f'    <MDI key="X_BAND">1</MDI>\n'
            f'    <MDI key="Y_DATASET">{lat_tif}</MDI>\n'
            f'    <MDI key="Y_BAND">1</MDI>\n'
            f'    <MDI key="PIXEL_OFFSET">0</MDI>\n'
            f'    <MDI key="LINE_OFFSET">0</MDI>\n'
            f'    <MDI key="PIXEL_STEP">1</MDI>\n'
            f'    <MDI key="LINE_STEP">1</MDI>\n'
            f'    <MDI key="SRS">EPSG:4326</MDI>\n'
            f'    <MDI key="GEOREFERENCING_CONVENTION">PIXEL_CENTER</MDI>\n'
            f'  </Metadata>\n'
            f'{band_xml}\n'
            f'</VRTDataset>'
        )

        vrt_path = os.path.join(tmpdir, 'data.vrt')
        with open(vrt_path, 'w') as f:
            f.write(vrt_xml)

        # Warp
        warped_tif = os.path.join(tmpdir, 'warped.tif')
        print(f'[warp] gdal.Warp ({nbands} bands)...')
        result = gdal.Warp(warped_tif, vrt_path, options=gdal.WarpOptions(
            format='GTiff', dstSRS='EPSG:4326',
            outputBounds=(min_lon, min_lat, max_lon, max_lat),
            xRes=res, yRes=res,
            resampleAlg='bilinear', geoloc=True,
            errorThreshold=0.125,
            srcNodata=-9999.0, dstNodata=np.nan,
            multithread=True,
            creationOptions=['COMPRESS=NONE', 'BIGTIFF=IF_SAFER'],
        ))
        if result is None:
            print('[warp] gdal.Warp returned None!')
            return None, None, None

        gt = result.GetGeoTransform()
        proj = result.GetProjection()
        print(f'[warp] Output: {result.RasterXSize} x {result.RasterYSize}, GT={gt}')
        warped = [result.GetRasterBand(i + 1).ReadAsArray().astype(np.float32)
                  for i in range(nbands)]
        result = None
        return warped, gt, proj

    except Exception as e:
        print(f'[warp] Error: {e}')
        import traceback; traceback.print_exc()
        return None, None, None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def corner_based_gt(lat, lon, th, tw):
    """Strategy 4: approximate geotransform from lat/lon extent."""
    valid = (np.abs(lat) <= 90) & (np.abs(lon) <= 360) & np.isfinite(lat) & np.isfinite(lon)
    if not np.any(valid):
        return None, None
    min_lon, max_lon = float(np.min(lon[valid])), float(np.max(lon[valid]))
    min_lat, max_lat = float(np.min(lat[valid])), float(np.max(lat[valid]))
    gt = (min_lon, (max_lon - min_lon) / tw, 0.0,
          max_lat, 0.0, -(max_lat - min_lat) / th)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    print(f'[crs:4] Corner-based GT: {gt}')
    return gt, srs.ExportToWkt()


def patch_envi_header(hdr_path, gt, proj_wkt):
    """Ensure ENVI .hdr has map info + coordinate system string."""
    if gt is None or gt == (0, 1, 0, 0, 0, 1):
        return

    ref_x = gt[0] + gt[1] / 2.0
    ref_y = gt[3] + gt[5] / 2.0
    x_size = abs(gt[1])
    y_size = abs(gt[5])

    map_info = (f"map info = {{Geographic Lat/Lon, 1, 1, "
                f"{ref_x:.10f}, {ref_y:.10f}, "
                f"{x_size:.10f}, {y_size:.10f}, "
                f"WGS-84, units=Degrees}}")
    coord_sys = f"coordinate system string = {{{proj_wkt}}}" if proj_wkt else None

    with open(hdr_path, 'r') as f:
        content = f.read()

    changed = False
    if 'map info' not in content.lower():
        content += '\n' + map_info + '\n'
        changed = True
        print('[hdr] Added map info')
    elif 'Geographic Lat/Lon' not in content:
        lines = [l for l in content.split('\n') if 'map info' not in l.lower()]
        lines.append(map_info)
        content = '\n'.join(lines) + '\n'
        changed = True
        print('[hdr] Replaced map info')

    if coord_sys and 'coordinate system string' not in content.lower():
        content += coord_sys + '\n'
        changed = True
        print('[hdr] Added coordinate system string')

    if changed:
        with open(hdr_path, 'w') as f:
            f.write(content)


def netcdf_to_envi(nc_file, output_file, geo_file=None, bands_only=False,
                    pattern=None, target_res=None, no_warp=False):
    """
    Convert a single netCDF file to georeferenced ENVI Type-4 (Float32).
    Returns True if georeferencing succeeded.
    """
    print(f'\n{"="*70}')
    print(f'  Input:  {nc_file}')
    print(f'  Output: {output_file}')
    if geo_file:
        print(f'  Geo:    {geo_file}')
    print(f'{"="*70}')

    subdatasets = get_subdatasets(nc_file)
    if not subdatasets:
        subdatasets = [(nc_file, os.path.basename(nc_file))]

    print(f'\n[info] {len(subdatasets)} subdatasets:')
    for i, (n, d) in enumerate(subdatasets, 1):
        info = sd_info(n)
        sz = f"{info['w']}x{info['h']}" if info else "?"
        print(f'  {i:3d}. [{sz:>10s}] {d}')

    tw, th = finest_dims(subdatasets)
    if tw == 0:
        print('[error] Cannot determine dimensions.')
        return False
    print(f'\n[info] Target: {tw} x {th}')

    # Strategy 1: native geotransform
    gt, proj = try_native_gt(subdatasets)

    # Strategy 2: CF-convention
    if gt is None:
        gt, proj = try_cf_coords(subdatasets)

    # Read bands
    print('\n[info] Reading bands...')
    all_bands = []
    band_names = []

    try:
        from scipy.ndimage import zoom as _z
        has_scipy = True
    except ImportError:
        has_scipy = False
        print('[warn] scipy missing — mismatched bands skipped')

    for sd_name, sd_desc in subdatasets:
        if bands_only and ('quality_flags' in sd_desc.lower() or 'uncert_index' in sd_desc.lower()):
            continue
        if pattern and not re.search(pattern, sd_desc, re.IGNORECASE):
            continue

        ds = gdal.Open(sd_name)
        if ds is None:
            continue

        w, h = ds.RasterXSize, ds.RasterYSize
        var_name = sd_name.split(':')[-1].strip().replace('/', '_').lstrip('_')

        for bi in range(1, ds.RasterCount + 1):
            data = ds.GetRasterBand(bi).ReadAsArray()
            if data is None:
                continue

            md = ds.GetRasterBand(bi).GetMetadata()
            md.update(ds.GetMetadata())
            data = data.astype(np.float64)

            fill = md.get('_FillValue')
            if fill is not None:
                try: data[data == float(fill)] = np.nan
                except (ValueError, TypeError): pass

            scale = md.get('scale_factor') or md.get('Scale')
            offset = md.get('add_offset') or md.get('Offset')
            if scale is not None:
                try: data *= float(scale)
                except (ValueError, TypeError): pass
            if offset is not None:
                try: data += float(offset)
                except (ValueError, TypeError): pass

            if (h, w) != (th, tw):
                if has_scipy:
                    data = resample_2d(data, th, tw)
                else:
                    continue

            bname = f"{var_name}_b{bi}" if ds.RasterCount > 1 else var_name
            all_bands.append(data.astype(np.float32))
            band_names.append(bname)

        ds = None

    print(f'[info] {len(all_bands)} bands collected')
    if not all_bands:
        print('[error] No bands!')
        return False

    # Strategy 3: geolocation warp
    if gt is None and geo_file and not no_warp:
        print(f'\n[crs:3] Warping via {os.path.basename(geo_file)}')
        lat, lon = read_latlon(geo_file)
        if lat is not None:
            if lat.shape != (th, tw):
                print(f'[geo] Resampling lat/lon {lat.shape} → ({th},{tw})')
                lat = resample_2d(lat, th, tw)
                lon = resample_2d(lon, th, tw)
            warped, gt_w, proj_w = warp_bands_via_geolocation(
                all_bands, lat, lon, th, tw, target_res)
            if warped is not None:
                all_bands, gt, proj = warped, gt_w, proj_w
                th, tw = all_bands[0].shape
                print(f'[crs:3] Warp OK → {tw} x {th}')

    # Strategy 4: corner fallback
    if gt is None and geo_file:
        print(f'\n[crs:4] Corner-based fallback')
        lat, lon = read_latlon(geo_file)
        if lat is not None:
            if lat.shape != (th, tw):
                lat = resample_2d(lat, th, tw)
                lon = resample_2d(lon, th, tw)
            gt, proj = corner_based_gt(lat, lon, th, tw)

    # Write ENVI
    print(f'\n[write] {tw} x {th}, {len(all_bands)} bands, Float32')
    driver = gdal.GetDriverByName('ENVI')
    out = driver.Create(output_file, tw, th, len(all_bands), gdal.GDT_Float32)
    if gt:
        out.SetGeoTransform(gt)
    if proj:
        out.SetProjection(proj)

    for i, (data, name) in enumerate(zip(all_bands, band_names), 1):
        band = out.GetRasterBand(i)
        band.WriteArray(data)
        band.SetDescription(name)
        band.SetNoDataValue(np.nan)
        band.FlushCache()

    out.FlushCache()
    out = None

    # Patch header
    hdr_path = output_file.rsplit('.', 1)[0] + '.hdr'
    if os.path.exists(hdr_path) and gt:
        patch_envi_header(hdr_path, gt, proj)

    # Verify
    print(f'\n{"="*70}')
    print(f'  VERIFICATION')
    print(f'{"="*70}')
    vds = gdal.Open(output_file)
    if vds:
        vgt = vds.GetGeoTransform()
        vpr = vds.GetProjection()
        gt_ok = vgt is not None and vgt != (0, 1, 0, 0, 0, 1)
        pr_ok = vpr is not None and vpr != ''
        print(f'  Size:       {vds.RasterXSize} x {vds.RasterYSize}')
        print(f'  Bands:      {vds.RasterCount}')
        print(f'  GeoTransform: {"OK" if gt_ok else "MISSING"}')
        print(f'  Projection:   {"OK" if pr_ok else "MISSING"}')
        if gt_ok:
            ulx, uly = vgt[0], vgt[3]
            lrx = vgt[0] + vgt[1] * vds.RasterXSize
            lry = vgt[3] + vgt[5] * vds.RasterYSize
            print(f'  UL: ({ulx:.6f}, {uly:.6f})')
            print(f'  LR: ({lrx:.6f}, {lry:.6f})')
            print(f'  Pixel: {abs(vgt[1]):.6f} x {abs(vgt[5]):.6f} deg')
        if pr_ok:
            s = osr.SpatialReference(wkt=vpr)
            print(f'  CRS: {s.GetName()} (EPSG:{s.GetAuthorityCode(None) or "?"})')
        vds = None

    if os.path.exists(hdr_path):
        print(f'\n[hdr] {os.path.basename(hdr_path)}:')
        with open(hdr_path) as f:
            for line in f:
                print(f'  {line.rstrip()}')

    status = "OK" if (gt and gt != (0, 1, 0, 0, 0, 1)) else "MISSING"
    print(f'\n{"="*70}')
    print(f'  {output_file}  [georef: {status}]')
    print(f'{"="*70}')
    return status == "OK"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: CLI — UNIFIED INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Download VIIRS data from LAADS DAAC and convert to georeferenced ENVI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODE 1: DOWNLOAD + CONVERT
  Download VNP02IMG + VNP03IMG for a date range, convert each to ENVI:

  python3 %(prog)s download \\
      --start 20250901 --end 20250903 \\
      --token YOUR_EARTHDATA_TOKEN \\
      --dir /data/viirs \\
      --region Canada \\
      --bands-only

MODE 2: CONVERT ONLY
  Convert an existing netCDF to ENVI (geo file auto-detected or specified):

  python3 %(prog)s convert \\
      --input VNP02IMG.A2025245.0000.002.*.nc \\
      --output output.bin \\
      --geo VNP03IMG.A2025245.0000.002.*.nc \\
      --bands-only

  python3 %(prog)s convert \\
      --input VNP02IMG.nc --output out.bin \\
      --pattern "I0[1-3]" --resolution 0.004
""")

    sub = p.add_subparsers(dest='mode', help='Operating mode')

    # ── download subcommand ──
    dl = sub.add_parser('download', help='Download from LAADS DAAC + convert')
    dl.add_argument('--start', required=True,
                    help='Start date YYYYMMDD (inclusive)')
    dl.add_argument('--end', required=True,
                    help='End date YYYYMMDD (exclusive)')
    dl.add_argument('--token', required=True,
                    help='LAADS DAAC / Earthdata Bearer token')
    dl.add_argument('--dir', required=True,
                    help='Base directory for downloads and output')
    dl.add_argument('--region', default=None,
                    help='Administrative area name (e.g. "Canada")')
    dl.add_argument('--img-product', default='VNP02IMG',
                    help='Radiance product (default: VNP02IMG)')
    dl.add_argument('--geo-product', default='VNP03IMG',
                    help='Geolocation product (default: VNP03IMG)')
    dl.add_argument('--bands-only', action='store_true',
                    help='Exclude quality_flags / uncert_index')
    dl.add_argument('--pattern', help='Regex filter for subdataset names')
    dl.add_argument('--resolution', type=float,
                    help='Force output pixel size in degrees')
    dl.add_argument('--no-warp', action='store_true',
                    help='Skip geolocation warp')
    dl.add_argument('--no-convert', action='store_true',
                    help='Download only, skip ENVI conversion')
    dl.add_argument('--workers', type=int, default=1,
                    help='Parallel download workers (default: 1)')

    # ── convert subcommand ──
    cv = sub.add_parser('convert', help='Convert existing netCDF to ENVI')
    cv.add_argument('--input', required=True, help='Input netCDF file')
    cv.add_argument('--output', required=True, help='Output ENVI .bin file')
    cv.add_argument('--geo', help='Geolocation file (VNP03IMG*.nc)')
    cv.add_argument('--bands-only', action='store_true')
    cv.add_argument('--pattern', help='Regex filter for subdataset names')
    cv.add_argument('--resolution', type=float,
                    help='Force output pixel size in degrees')
    cv.add_argument('--no-warp', action='store_true')

    args = p.parse_args()

    if args.mode is None:
        p.print_help()
        sys.exit(1)

    # Check scipy
    try:
        import scipy
    except ImportError:
        print('[error] scipy required: pip install scipy')
        sys.exit(1)

    if args.mode == 'convert':
        # ── Convert mode ──
        if not os.path.exists(args.input):
            print(f'[error] Not found: {args.input}')
            sys.exit(1)

        geo = args.geo
        if geo and not os.path.exists(geo):
            print(f'[error] Geo file not found: {geo}')
            sys.exit(1)
        if not geo:
            geo = find_local_geo_file(args.input)
        if not geo:
            print('\n' + '!' * 70)
            print('  WARNING: No geolocation file!')
            print('  VNP02IMG has NO embedded lat/lon — you need VNP03IMG.')
            print('  Use --geo VNP03IMG.nc or place it in the same directory.')
            print('!' * 70 + '\n')

        ok = netcdf_to_envi(
            args.input, args.output,
            geo_file=geo,
            bands_only=args.bands_only,
            pattern=args.pattern,
            target_res=args.resolution,
            no_warp=args.no_warp,
        )
        sys.exit(0 if ok else 1)

    elif args.mode == 'download':
        # ── Download mode ──
        start = datetime.datetime.strptime(args.start, '%Y%m%d')
        end = datetime.datetime.strptime(args.end, '%Y%m%d')
        if end <= start:
            # Single day: make end = start + 1
            end = start + datetime.timedelta(days=1)

        days = []
        dt = start
        while dt < end:
            days.append(dt)
            dt += datetime.timedelta(days=1)

        print(f'\n[plan] {len(days)} day(s): {start.strftime("%Y-%m-%d")} → {end.strftime("%Y-%m-%d")}')
        print(f'[plan] Products: {args.img_product} + {args.geo_product}')
        print(f'[plan] Region: {args.region or "global"}')
        print(f'[plan] Dir: {args.dir}')

        total_pairs = []

        def process_day(dt):
            pairs = download_viirs_pair(
                dt, args.dir, args.token,
                regions=args.region,
                img_product=args.img_product,
                geo_product=args.geo_product,
            )
            return pairs

        if args.workers > 1 and len(days) > 1:
            with Pool(args.workers) as pool:
                results = pool.map(process_day, days)
            for r in results:
                total_pairs.extend(r)
        else:
            for dt in days:
                total_pairs.extend(process_day(dt))

        print(f'\n[download] Total: {len(total_pairs)} granule pairs downloaded')

        if args.no_convert:
            print('[done] Skipping conversion (--no-convert)')
            sys.exit(0)

        # Convert each pair
        success = 0
        fail = 0
        for img_path, geo_path in total_pairs:
            out_bin = img_path.rsplit('.', 1)[0] + '.bin'
            ok = netcdf_to_envi(
                img_path, out_bin,
                geo_file=geo_path,
                bands_only=args.bands_only,
                pattern=args.pattern,
                target_res=args.resolution,
                no_warp=args.no_warp,
            )
            if ok:
                success += 1
            else:
                fail += 1

        print(f'\n{"="*70}')
        print(f'  SUMMARY: {success} converted, {fail} failed')
        print(f'{"="*70}')
        sys.exit(0 if fail == 0 else 1)


if __name__ == '__main__':
    main()
