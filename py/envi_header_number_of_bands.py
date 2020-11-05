# output the number of bands, from an envi header
import sys
args = sys.argv

if len(args) < 2:
    print("python3 envi_header_number_of_bands.py [envi .hdr file] # list number of bands in .hdr file")
    sys.exit(1)

lines = open(args[1]).readlines()

for i in range(0, len(lines)):
    line = lines[i].strip()

    w = line.split("=")
    if len(w) > 1:
        if w[0].strip() == 'bands':
            nb = int(w[1].strip())

print(nb)
