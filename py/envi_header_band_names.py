# print out band names from envi header file # check if this is redundant with misc.py
import os
import sys
args = sys.argv

if len(args) < 2:
    print("python3 envi_header_band_names [envi .hdr file] # print band names within envi hdr file")

lines, bandname_lines = open(args[1]).readlines(), []
n_band_names, in_band_names = 0, False

for i in range(0, len(lines)):
    line = lines[i].strip()

    if len(line.split("band names =")) > 1:
        in_band_names = True

    if in_band_names:
        n_band_names += 1
        if len(line.split("}")) < 2:
            w = line.split(',')
            line = ''.join(w[:-1]) + ',' + w[-1]
            lines[i] = line
        else: # on last band names line:
            lines[i] = line.replace(',', '')

    if in_band_names:
        bandname_lines.append(line) # track band names we have
    else:
        pass
        # non_bandname_lines.append(line) # record non-band-name lines,
    if in_band_names:
        if len(line.split("}")) > 1:
            in_band_names = False

bandname_lines[0] = bandname_lines[0].split('{')[1]
bandname_lines[-1] = bandname_lines[-1].strip('}')
bandname_lines = [x.strip(',').strip() for x in bandname_lines]

for b in bandname_lines:
    print(b)
