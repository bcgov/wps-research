## Data Engineering
* Have a look at p 1-2 of [CSRS abstracts](https://github.com/bcgov/wps-research/blob/master/doc/2024_csrs/2024_csrs_abstracts.pdf)
* Review general details of [Sentinel-2 mission](https://sentiwiki.copernicus.eu/web/s2-mission)
* Install google cloud sdk [install google cloud SDK](https://cloud.google.com/sdk/docs/install). Could borrow the script [here](https://github.com/bcgov/wps-research/blob/master/py/gcp/install_gcp.py) and update it to the latest versions (and to work on MacOS)
* Install QGis
* Familiarize with Sentinel-2 tiling grid [https://sentiwiki.copernicus.eu/web/s2-products](https://sentiwiki.copernicus.eu/web/s2-products) by opening in QGis. Can add an XYZ layer (e.g. OpenStreetMap) to see the grid's relation to some geographic features
* Note: the grid is also available in Shapefile, clipped to BC area [here](https://github.com/bcgov/wps-research/blob/master/py/sentinel2_bc_tiles_shp/Sentinel_BC_Tiles.shp)
* Write a python function (in a .py file) to download all Sentinel-2 data (from GCP) available, in a time window (yyyymmdd1, yyyymmdd2) for one grid location e.g. T10UFB is Kamloops. Hopefully could reuse some of: [this one](https://github.com/bcgov/wps-research/blob/master/py/gcp/update_tile.py) which uses [gsutil rsync](https://cloud.google.com/storage/docs/gsutil/commands/rsync)
* Review BC documents [here](https://www2.gov.bc.ca/assets/gov/farming-natural-resources-and-industry/forestry/stewardship/forest-analysis-inventory/data-management/news/burn_severity_mapping_summary_210823.pdf) and [here](https://www2.gov.bc.ca/assets/gov/farming-natural-resources-and-industry/forestry/stewardship/forest-analysis-inventory/data-management/news/wildfire_2023_burn_severity_and_high_resolution_imagery.pdf) (should be a similar document for 2022 as well)
* Verify this is the correct link and download a province-wide burned-severity dataset for 2021 [here](https://catalogue.data.gov.bc.ca/dataset/fire-burn-severity-historical). Open it in QGis : )  
* Determine if "pre" and "post" imagery dates (used to generate the product) are listed within the dataset
* Download 2021 fire perimeters from: [here](https://www.for.gov.bc.ca/ftp/HPR/external/!publish/Maps_and_Data/GoogleEarth/WMB_Fires/) in KML format
* Fire of interest: Sparks lake K21001. Note: pre/post dates used in BC Gov BS estimate: 20200729 / 20220902  
* Rasterize burned severity product 
## Modelling
* Fit a sequence of models: where the independent variable is a time-series of Sentinel-2 data (starting with a cloud-free pre-fire date, and ending with date "X") Where "X" is >= the pre-fire date, and "X" <= the post-fire date. The post-fire date is the first cloud-free date after the fire is declared "out" (should be available from national fire polygon database, if not [here](https://www.for.gov.bc.ca/ftp/HPR/external/!publish/Maps_and_Data/GoogleEarth/WMB_Fires/). Easier: can choose a post-fire date by inspection (some time late in the season when the fire has obviously stopped moving).      
* Two cases: dependent variable is 1) burned-severity class or 2) the dNBR.
* Want to understand the goodness of fit for the dependent variable, as "X" is varied (want to see how small we can make "X" and still get a good estimate).
* Methods: start with [Scikit-learn](https://scikit-learn.org/stable/) and find something that runs in a finite amount of time. Try a few models in scikit-learn at least, before moving to more complex neural-network models such as in Pytorch or Keras/[Tensorflow](https://developers.google.com/machine-learning/crash-course)
### Preliminaries
* Select data frames (betwee pre-fire and post-fire date) with some cloud cover threshold e.g. <= 7%
* Likely want to plot the "spectra" (one curve for each band, date on the X-axis) for a variety of points: unburned, and different burned severity classes to get an idea of how the values change. Could write a script that does this using matplotlib. Can add coordinates for some manually selected points (of various classes) at the top of the file.   
## To consider later
* As a future refinement, may likely need to run Sen2cor processor [here](https://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor-v2-11/) as a pre-processing step to exclude detected areas. Info available [here](https://sentiwiki.copernicus.eu/web/s2-processing#S2Processing-L2AAlgorithmsS2-Processing-L2A-Algorithmstrue) on S2 processing algorithms resulting in the available cloud mask accompanying Level-2 data (running sen2cor on Level-1 data results in Level-2 data) 
* It may eventually be necessary to improve cloud vs. smoke vs. fire classification to refine our results.
## steps for Application-izing for fire and burned-severity mapping
1. User provides fire number
2. Pull current bc gov fire perimeter and point (shapefiles) 
3. Extract the perimeter and/or point for that shapefile (default to perimeter, if available) 
4. Calculate an AOI around the perimeter/point (e.g. take a "bounding box" around the perimeter/ point and "make it bigger") 
5. Intersect the new AOI with the sentinel-2 tile grid. i.e., data/Sentinel_BC_Tiles.tar.gz to determine the relevant tile-ID to download (e.g. T10UFB would be included for a fire near kamloops) 
6. Download (default to 6 weeks for fire mapping application) of data up to present day, over all the selected tile-ID using sync_daterange_gid_zip.py. Can delete zip files at this point to save space. For burn-severity mapping, might need much more to get a pre-image). In the fire-mapping case, if there are present imagery available that are actually cloud-free, can skip compositing / multi-date stuff and just pull the present imagery
7. Extract cloud-free swir/nir data for all downloaded frames, using sentinel2_extract_cloudfree_swir_nir.py
8. run sentinel2_mrap.py to produce a series of composites for each tile-ID  
9. run sentinel2_mrap_merge.py. Note: may need to modify to only spit out a result for the last possible step (most recent data) only, otherwise will run out of space.  However, might need results at intermediary days if the results are funky . Also might need all steps for burned-severity case.
10. Clip merged composite(s) to AOI 
11. Run your stuff to produce a BARC map  (pre/post dates). Fire mapping case: could just default to BARC mapping (if the data are not too heavy) but collapse the result down to a single class. If there are data issues, could apply "red wins" rule to (R,G,B) = (B12, B11, B9) data (present imagery or composite) 
12. Fully application-ised version might default to using BARC mapping to generate a fire map (binary classification). However provided some alternate steps/ work-arounds forcases where the data volume gets impractical. 


