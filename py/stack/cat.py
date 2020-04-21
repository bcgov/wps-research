'''
draft high level interface for stacking

this script asserts geo and other information match, as output from 
http://nicolas.brodu.net/recherche/sen2res/

and collets band names from the output header files'''
import os
import sys

files = os.popen("ls -1 *output4.bin").readlines()
files = [f.strip() for f in files]

first_lines = None
first_linesall = None

for f in files:
    # print(f)

    hf = f[:-3] + "hdr"
    # print(hf)

    lines = open(hf).readlines()
    lines = [line.strip() for line in lines]

    first = lines[:-12]
    first_linesall = lines

    if first_lines is None:
        first_lines = first
        for i in range(0, len(first_lines)):
            print(lines[i])
    else:
        for i in range(0, len(first_lines)):
            if first[i] != first_lines[i]:
                print("Error: mismatch")
                print("\t", first_lines[i])
                print("\t", first[i])
        print(lines[len(first_lines)])
        for i in range(0, len(lines)):
            if lines[i] != first_linesall[i]:
                print("Error: mismatch2")
                print(lines[i])
                print(first_lines_all[i])

    print("--")
    import copy
    band_lines = copy.deepcopy(first_linesall[13:])

    # MSIL1C_20191129T190741_N0208_R013_T10UFB_20191129T204451_S2A_
    w = f.split("_")[:-1]
    # w = w[1:]
    # print(w)
    for i in range(1, len(band_lines)):
        line = band_lines[i]
        # "B4 (665 nm),"
        line = line.replace(" (", "_")
        line = line.replace(" ", "_")
        line = line.replace(")", "")
        line = "_".join(w) + "_" + line
        line = line.strip("}")
        band_lines[i] = line
        print(line)

