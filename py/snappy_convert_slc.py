#!/usr/bin/python2
''' given a collection of unzipped RCM Quad-pol data: for each SLC folder:
        1) convert to PolSARPro format ("i/q format"
        2) convert i/q format to s2 "Scattering Matrix" format
        3) apply scm method (a visualization method of Shane Cloude)

references:
    [1] https://senbox.atlassian.net/wiki/spaces/SNAP/pages/19300362/How+to+use+the+SNAP+API+from+Python '''

import os # usual python stuff
import sys
import snappy # ESA SNAP stuff
from snappy import ProductIO

def md(d):  # make folder if not yet exist
    if not os.path.exists(d): os.mkdir(d)

exe = "/home/" + os.popen("whoami").read().strip() + "/GitHub/wps-research/cpp/convert_iq_to_s2.exe"
exe = os.path.abspath(exe)
print(exe)

if not os.path.exists(exe):
    print("Error: failed to find convert_iq_to_s2.exe"); sys.exit(1)

files = [f.strip() for f in os.popen("ls -1d RCM_SLC_ZIP/*_SLC").readlines()]
folders = [f + '/' for f in files]

print(folders)
for i in range(0, len(folders)):
    f = folders[i] + 'manifest.safe'; print(f)

    out_folder = folders[i] + 'PolSARPro'; md(out_folder)
    s2_folder = folders[i]  + 's2'; md(s2_folder)
    scm_folder = folders[i] + 'scm'; md(scm_folder)

    d = ProductIO.readProduct(f)  # read SLC data 
    ProductIO.writeProduct(d, out_folder, 'PolSARPro')  # convert to PolSARPro format

    a = os.system(exe + " " + out_folder + " " + s2_folder)
    a = os.system(' '.join(['scm',  # a visualization method of Shane Cloude
                            s2_folder, # input s2 matrix format data
                            scm_folder, # output for visualization result
                            'box', # filter type used for visualization
                            '5', # filter window size
                            'yes', # correct for average faraday rotation on scene
                            '1', # alpha angle used for visualization
                            '3'])) # vertical multiook factor

    f1 = folders[i] + 'scm_fr'
    f2 = folders[i] + 'scm_fr_t4'
    f3 = folders[i] + 'scm_fr_t4_fl'
    for fi in [f1, f2, f3]:
        cmd = 'rm -rf ' + fi
        print(cmd)
        a = os.system(cmd)
