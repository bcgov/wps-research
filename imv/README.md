# imv

Entry point for BCWS FTL MVP software
* [Please click here for setup instructions for BCWS FTL MVP software](https://github.com/bcgov/bcws-psu-research/blob/master/SETUP.md)

**Basic ML algorithm** included: 
* semi-supervised kmeans variant [Python based implementation](https://github.com/bcgov/bcws-psu-research/blob/master/py/kmeans_optimize.py) with C/C++ implementations under the hood:
* [nearest centres algorithm](https://github.com/bcgov/bcws-psu-research/blob/master/cpp/raster_nearest_centre.cpp) followed by:
* [kmeans iteration](https://github.com/bcgov/bcws-psu-research/blob/master/cpp/kmeans_iter.cpp) 

# Sample data
## Chess sample data
[please click here to explore chess sample data](https://github.com/bcgov/bcws-psu-research/tree/master/imv/chess)

## Real chess sample data
[please click here to explore real chess sample data](https://github.com/bcgov/bcws-psu-research/tree/master/imv/chess_real)

## Peppers hyperspectral data
[please click here to explore hyperspectral peppers sample data](https://github.com/bcgov/bcws-psu-research/tree/master/imv/peppers)

## Bonaparte lake (Sentinel2) data
[please click here to explore Bonaparte Lake, BC Sentinel2 data](https://github.com/bcgov/bcws-psu-research/blob/master/imv/bonaparte/README.md)

# info

Imv short for "image viewer", also an anagram of vim: minimalist hyperspectral image viewer loosely inspired by ENVI and Warcraft II.

Raster data inputs always assumed to be: ENVI binary format, data type "4" aka IEEE-standard (byte-order 0) Float32, "band sequential" format, with certain restrictions on formatting of the header file

# operations (key combinations)
band switching by r[number]&lt;return>, g[number]&lt;return>, b[number]&lt;return>

## Example: default bands (r,g,b = 1,2,3) natural colour

![sample](rgb-1,2,3.png)

## Example: band selection (r,g,b = 4,3,2) so-called false colour

![sample](rgb-4,3,2.png)
