# imv

Entry point for BCWS FTL MVP software [please click here for setup instructions for BCWS FTL MVP software](https://github.com/bcgov/wps-research/blob/master/imv/SETUP.md)

**imv** short for "image viewer" (also an anagram of vim the unix text editor): **imv is a minimalist hyperspectral image viewer** and is loosely inspired by ENVI and Warcraft II and conceptually derived from an earlier program called scg.
* **will accept data size near to system limit e.g. exabyte scale on linux systems**
* contains a number of **multithreaded functions for high performance**
* interactive 3d scatter plots for overview and subscene windows
* big-data viewer with video game responsiveness
* tested on images **hundreds of GB in size**
* spectral plotting at point location
* resizeable target/zoom window

**Basic ML algorithm** included: 
* semi-supervised kmeans variant [Python based implementation](https://github.com/bcgov/wps-research/blob/master/py/kmeans_optimize.py) with C/C++ implementations under the hood:
* [nearest centres algorithm](https://github.com/bcgov/wps-research/blob/master/cpp/raster_nearest_centre.cpp) followed by:
* [kmeans iteration](https://github.com/bcgov/wps-research/blob/master/cpp/kmeans_iter.cpp) 

# Sample data

## Real chess sample data
[please click here to explore real chess sample data](https://github.com/bcgov/wps-research/tree/master/imv/chess_real)

## Synthetic chess sample data
[please click here to explore chess sample data](https://github.com/bcgov/wps-research/tree/master/imv/chess)

## Peppers hyperspectral data
[please click here to explore hyperspectral peppers sample data](https://github.com/bcgov/wps-research/tree/master/imv/peppers)

## Bonaparte lake (Sentinel2) data
[please click here to explore Bonaparte Lake, BC Sentinel2 data](https://github.com/bcgov/wps-research/blob/master/imv/bonaparte/README.md)

# info

Raster data inputs always assumed to be: ENVI binary format, data type "4" aka IEEE-standard (byte-order 0) Float32, "band sequential" format, with certain restrictions on formatting of the header file

# operations (key combinations)
### click on scene window area, to buffer data
### band switching
band switching by r[number]&lt;return>, g[number]&lt;return>, b[number]&lt;return>
### Adjust histogram scaling
p xx 
percent
### move bands forward / back by one
### move bands forward / back a full date
### add annotation point
### delete annotation point
### list annotation points
### run K-means semi-supervised algorithm on whole scene
### run K-means semi-supervised algorithm on sub-scene
### change image-scaling method (s key)
### resize zoom window

Note: don't attempt to resize the other windows (should be disabled, is not)

### Example: default bands (r,g,b = 1,2,3) natural colour

![sample](rgb-1,2,3.png)

### Example: band selection (r,g,b = 4,3,2) so-called false colour

![sample](rgb-4,3,2.png)

# formats
## input raster format
## output raster format
## target / vector / annotation file format
Tolerant of extra col's. Note the row/col convention. Confirm data upload to LAN. Confirm can open output in QGIS. Compare to BC FTL
