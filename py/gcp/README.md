* update_tile.py the present downloading utility.
* instruction delete index.csv.gz to refresh?
* need to add instructions for installing gcp, here
* instruction on command used: [rsync command on gcp](https://cloud.google.com/storage/docs/gsutil/commands/rsync)
* need instruction on general use:

'''
 python3 ~/GitHub/wps-research/py/gcp/update_tile.py T10UFB 100.  20210626 20210815 
 python3 ~/GitHub/wps-research/py/gcp/run_sen2cor.py 
 python3 ~/GitHub/wps-research/py/sentinel2_stack_all.py  1 
 python3 ~/GitHub/wps-research/py/sentinel2_swir_subselect_all.py
 python3 ~/GitHub/wps-research/py/sentinel2_swir_resample.py 
 python3 ~/GitHub/wps-research/py/sentinel2_swir20_dominant.py
# now need to add in script to replace B10!
'''
