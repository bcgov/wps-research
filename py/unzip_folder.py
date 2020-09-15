# extract all zips in one folder, to another (destination) folder
import os
import sys
args = sys.argv

def err(m):
    print("Error: " + m); sys.exit(1)

if len(args) != 3: err("Usage: unzip_folder [input folder containing zips] [output dest folder]")

sep = os.path.sep
i_d = os.path.abspath(args[1]) + sep
o_d = os.path.abspath(args[2]) + sep

if not os.path.exists(i_d): err("input directory doesn't exist")
if not os.path.exists(o_d): err("output directory doesn't exist")

zips = os.popen("ls -1 " + i_d + "*.zip").readlines()
zips = [z.strip() for z in zips]

for z in zips:
    cmd = "unzip -o -d " + o_d + " " + z
    print(cmd)
    a = os.system(cmd)
