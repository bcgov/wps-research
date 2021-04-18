

```
ogr2ogr -sql "SELECT ST_Centroid(geometry), * FROM Sentinel_BC_Tiles" -dialect sqlite centroid.shp Sentinel_BC_Tiles.shp
```
