# nrtbs (Near-Real-Time Burned Severity)
Application: **NRT "same-day" burned severity (automatic)**

* uses MRAP (Most Recent Available Pixel) "cloud-free" image compositing
* Access to ESA Sentinel-2 data via NRCAN NRT Sentinel products mirror on AWS-S3 (thanks ESA and NRCAN) 
 
## To run
Simple option:
```
wget https://github.com/bcgov/nrtbs/archive/refs/heads/master.zip
unzip master.zip
cd nrtbs-master
python3 py/get_composite [FIRE_NUMBER]
```

Advanced option (if you have github set up already at your terminal):
```
git clone git@github.com:bcgov/nrtbs.git
cd nrtbs
python3 py/get_composite [FIRE_NUMBER] 
```
where [FIRE_NUMBER] is a 6-character BC wildfire "fire number" (a letter followed by 5 digits), for example:
```
python3 py/get_composite.py G90267
```
for the 2024 Parker Lake wildfire, affecting Fort Nelson (BC). 

To manually add an end date, prefix the fire number accordingly:
```
python3 py/get_composite.py 20240630 G90267
```
* Start and end dates will be automatically generated based on fire ignition dates unless manual end date is given
* Automatic trimming to fire AOI
* Results will be output into a FIRE_NUMBER_barcs folder
## Notes:
### Skipping refreshing Sentinel-2 data index and/or skipping download of Sentinel-2 data
Can add flag ```--no_update_listing``` to skip refreshing the index of all available Sentinel-2 data. Also a ```--skip_download``` is available for re-running without downloading the initial .zip format data again (e.g. if there are storage limitations, can use this to re-run after deleting all zip files but keeping intermediary products)
e.g.:
```
python3 py/get_composite.py G90267 --no_update_listing
```
### Historical mode
For research purposes it's important to be able to generate composite imagery sequences over fires from past years. Therefore we've added an option to provide historical BC Wildfire information.

How to add historical perimeters and study a past fire that's not available in the default (current public) source of fire perimter data (e.g. insert your fire number of interest and path to your shapefile here):
```
python3 py/get_composite.py V82991 --historical_perimeters=data/BC_Fire_Perimeters_2023-polygon.shp --historical_points=data/BC_Fire_Perimeters_2023-point.shp
```

## Dependencies
* Windows: first [please click here for instructions to install WSL prompt](https://learn.microsoft.com/en-us/windows/wsl/install) no admin privileges required in Windows
* Ubuntu Linux
In both cases, the following commands are needed before running the application
```
sudo apt update && sudo apt upgrade
sudo apt install gdal-bin libgdal-dev python3-gdal python3-pip python3-tk
python3 -m pip install numpy matplotlib pandas rasterio geopandas awscli --break-system-packages
```
* Also compatible with MacOS (use brew install instead of sudo apt install) 

Note: may need to fix numpy at version 1.23.0 e.g. with:
```
python3 -m pip install numpy==1.23.0
```

### Test procedure:
To get started with the application (assuming you have WSL and dependencies installed) the available test procedure:
```
wget https://github.com/bcgov/nrtbs/archive/refs/heads/master.zip
unzip master.zip
cd nrtbs-master
python3 py/get_composite.py 20240601 G90267
```

# Background / references
* [Sterling von Dehn's work term report (internal, access controlled)](https://bcgov.sharepoint.com/:b:/t/01324/ERKaKW6P0AdPjP5yOgJofXEBFYHDg6iIm-7c1trylGTEuA?e=7iID0Z)
* [Sterling's original work instructions](doc/TASK.md)
* https://github.com/SashaNasonova/burnSeverity/blob/main/burnsev_gee.py
* https://github.com/SashaNasonova/burnSeverity/blob/main/BurnSeverityMapping.ipynb
* https://burnseverity.cr.usgs.gov/products/baer
* https://burnseverity.cr.usgs.gov/ravg/background-products-applications
* https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a
* [USDA Forest Service: Field Guide for Mapping Post-fire Soil Burn Severity](https://research.fs.usda.gov/treesearch/36236)
