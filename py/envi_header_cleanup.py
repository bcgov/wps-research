from misc import *

if len(args) < 2:
    err("python3 envi_header_cleanup.py [input envi header filename .hdr]")

if not exist(args[1]):
    err("file not found: " + args[1])

data = open(args[1]).read().strip()

data = data.replace("description = {\n", "description = {")
data = data.replace("band names = {\n", "band names = {")

lines = data.split("\n")

in_band_names = False
for i in range(0, len(lines)):
    line = lines[i].strip()

    if len(line.split("band names =")) > 1:
        in_band_names = True
    
    # print(line + (" TRUE" if in_band_names else ""))

    if in_band_names:
        if len(line.split("}")) < 2:
            w = line.split(',')
            line = ''.join(w[:-1]) + ',' + w[-1]
            lines[i] = line
        else: # on last band names line:
            lines[i] = line.replace(',', '')

    if in_band_names:
        if len(line.split("}")) > 1:
            in_band_names = False


print('\n'.join(lines))
