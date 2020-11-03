from misc import *

if len(args) < 2:
    err("python3 envi_header_cleanup.py [input envi header filename .hdr]")

if not exist(args[1]):
    err("file not found: " + args[1])

data = open(args[1]).read().strip()
n_band_names, in_band_names, nb = 0, False, 0
data = data.replace("description = {\n", "description = {")
data = data.replace("band names = {\n", "band names = {")
lines = data.split("\n")

for i in range(0, len(lines)):
    line = lines[i].strip()

    w = line.split("=")
    if len(w) > 1:
        if w[0].strip() == 'bands':
            nb = int(w[1].strip())

    if len(line.split("band names =")) > 1:
        in_band_names = True
    
    # print(line + (" TRUE" if in_band_names else ""))

    if in_band_names:
        n_band_names += 1
        if len(line.split("}")) < 2:
            w = line.split(',')
            line = ''.join(w[:-1]) + ',' + w[-1]
            lines[i] = line
        else: # on last band names line:
            lines[i] = line.replace(',', '')

    if in_band_names:
        if len(line.split("}")) > 1:
            in_band_names = False

data = ('\n'.join(lines)).strip()
print(data)

open(args[1] + '.bak', 'wb').write(open(args[1]).read().encode())
open(args[1], 'wb').write(data.encode())

print("number of bands", nb)
print("number of band names", n_band_names)
