''' SWIR subselection for all bands 20220407'''
import os
import sys
lines = [x.strip() for x in os.popen('find ./ -name "S*10m.bin"').readlines()]

for line in lines:
    cmd = '~/GitHub/bcws-psu-research/cpp/sentinel2_swir_subselect.exe ' + line
    print(cmd)
    a = os.system(cmd)
