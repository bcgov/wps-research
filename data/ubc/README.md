# Sentinel-2 (SWIR bands only) fire data
First example includes non-uniform smoke/haze. Second example has haze and some strong shadow at SouthEast corner.

## To run:
```
Rscript view.R
```

May need to 
```
install.packages('raster') 
```
first.

## Outputs:

|   | Linear scaling  | Histogram trimming  |
|---|---|---|
| G90292  | <img src="png/G90292_20230514.tif_scaled.png" width="650">  | <img src="png/G90292_20230514.tif_trimmed.png" width="650"> |
| G80223  | <img src="png/G80223_20230513.tif_scaled.png" width="650"> | <img src="png/G80223_20230513.tif_trimmed.png" width="650"> |

