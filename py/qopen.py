# 20230809 this script is not working yet 

from misc import args, err, run, exists
import os

if not exists(args[1]):
    err("please check input file: " + args[1])


# Check if QGIS is already running
x = os.popen("pgrep qgis").read().strip()


if x == '':  # Qgis is not already running
    print("Starting new instance of Qgis..")
    a = os.system("qgis " + args[1])
else:
    # QGIS is running, open data in existing instance
    q_code = 'import qgis.utils; qgis.utils.iface.addRasterLayer("' + args[1] + '","' + args[1] + '")'


    print(q_code)
    cmd = "qgis --nologo --code '" + q_code + "'"
    print(cmd)

