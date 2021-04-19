

```
ogr2ogr -sql "SELECT ST_Centroid(geometry), * FROM Sentinel_BC_Tiles" -dialect sqlite centroid.shp Sentinel_BC_Tiles.shp

python3 shapefile_to_csv.py sentinel2_bc_tiles_shp/centroid/centroid.shp
```
