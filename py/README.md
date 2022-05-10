# Programs description
## For producing normalized, sorted stacks of Sentinel2
* **gcp/update_tile.py** pull Level-1 Sentinel-2 data from Google Cloud Platform, over a given tile-id. MAX_CLOUD, DATE_MIN, DATE_MAX are parameters
* **gcp/run_sen2cor.py** run Sen2Cor (old version for backwards compatibility) on all Level-1 folders within present folder
* **sentinel2_stack_all.py** run this in a directory of Sentinel-2 L2 folders already processed by **gcp/run_sen2cor.py**

