'''add folders expected by sen2cor, if they are missing..why?
..Google cloud platform doesn't save empty folders

example: 1) fix folders to not confuse Sen2Cor 2) create a zip 3) run extractor to get integrated L1
  
  python3 ~/GitHub/wps-research/py/gcp/fix_s2.py S2A_MSIL1C_20220226T085931_N0400_R007_T35UQS_20220226T101548.SAFE
  zip -r S2A_MSIL1C_20220226T085931_N0400_R007_T35UQS_20220226T101548.zip S2A_MSIL1C_20220226T085931_N0400_R007_T35UQS_20220226T101548.SAFE
  python3 ~/GitHub/wps-research/py/sentinel2_extract_stack.py S2A_MSIL1C_20220226T085931_N0400_R007_T35UQS_20220226T101548.zip
'''
import os
import sys
sep = os.path.sep
my_path = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep
sys.path.append(my_path + "..")

from misc import args, sep, exists
f = args[1]

def md(f):
    if not exists(f):
        print('mkdir', f)
        os.mkdir(f)

md(f + sep + 'AUX_DATA')
md(f + sep + 'HTML')
