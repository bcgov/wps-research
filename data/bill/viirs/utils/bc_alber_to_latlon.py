from osgeo import gdal, osr

ds = gdal.Open("/home/bill/GitHub/wps-research/data/bill/C11659/aoi/pgfc_aoi.bin")
gt = ds.GetGeoTransform()
nx, ny = ds.RasterXSize, ds.RasterYSize

x_min = gt[0]
x_max = gt[0] + nx * gt[1]
y_max = gt[3]
y_min = gt[3] + ny * gt[5]

src_srs = osr.SpatialReference()
src_srs.ImportFromWkt(ds.GetProjection())
dst_srs = osr.SpatialReference()
dst_srs.ImportFromEPSG(4326)
ct = osr.CoordinateTransformation(src_srs, dst_srs)

corners = [(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)]
lons, lats = [], []
for x, y in corners:
    lat, lon, _ = ct.TransformPoint(x, y)
    lons.append(lon)
    lats.append(lat)

print(f"W: {min(lons):.6f}  E: {max(lons):.6f}")
print(f"S: {min(lats):.6f}  N: {max(lats):.6f}")