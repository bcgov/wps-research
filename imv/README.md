# imv

Entry point for BCWS FTL MVP software
[Please click here for setup instructions for BCWS FTL MVP software](https://github.com/bcgov/bcws-psu-research/blob/master/SETUP.md)

Basic ML algorithm included: 
*semi-supervised kmeans variant [Python based implementation](https://github.com/bcgov/bcws-psu-research/blob/master/py/kmeans_optimize.py) with
* [nearest centres algorithm](https://github.com/bcgov/bcws-psu-research/blob/master/cpp/raster_nearest_centre.cpp) and
* [kmeans iteration](https://github.com/bcgov/bcws-psu-research/blob/master/cpp/kmeans_iter.cpp) under the hood

# Sample data
## Chess sample data
[please click here to explore chess sample data](https://github.com/bcgov/bcws-psu-research/tree/master/imv/chess)

## Real chess sample data
[please click here to explore real chess sample data](https://github.com/bcgov/bcws-psu-research/tree/master/imv/chess_real)

## Peppers hyperspectral data
[please click here to explore hyperspectral peppers sample data](https://github.com/bcgov/bcws-psu-research/tree/master/imv/peppers)

#info

anagram of vim: "image viewer":

minimalist hyperspectral image viewer. band switching by r[number]&lt;return>, g[number]&lt;return>, b[number]&lt;return>

sample test data (ENVI binary format, data type "4" aka IEEE-standard (byte-order 0) Float32, "band sequential" format) red, green, blue, near-infrared

## Example: default bands (r,g,b = 1,2,3) natural colour

![sample](rgb-1,2,3.png)

## Example: band selection (r,g,b = 4,3,2) so-called false colour

![sample](rgb-4,3,2.png)
