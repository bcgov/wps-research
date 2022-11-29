'''20221128 a trusted spectra extraction, extracts multiple spectra using the _targets.csv file from imv'''
import os
import sys
import json
import struct
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import matplotlib.pyplot as plt
args = sys.argv

def err(m):
    print("Error: " + str(m)); sys.exit(1)

print(len(args))
if len(args) < 2:
    err("python3 extract_spectra.py [input image name]")

img = args[1] # input image
if not os.path.exists(img):
    err('file not found: ' + img)

tf = img + "_targets.csv"


lines = [x.strip() for x in open(tf).readlines()]
if lines[0] != 'feature_id,row,lin,xoff,yoff':
    err('expected: feature_id,row,lin,xoff,yoff')
lines = [x.split(',') for x in lines][1:]

pix_j, lin_j, label_j= [], [], []
for line in lines:
    pix_j += [int(line[1])]
    lin_j += [int(line[2])]
    label_j += [line[0]]

# pix_j = int(args[3]) # col 0-idx
# lin_j = int(args[2]) # row 0-idx

out_spec_fn = img + "_spectrum.csv"
out_spec_f = open(out_spec_fn, "wb")
print('+w', out_spec_fn)

Image = gdal.Open(img, gdal.GA_ReadOnly) # open image
nc, nr, nb = Image.RasterXSize, Image.RasterYSize, Image.RasterCount # rows, cols, bands

band_names = []
for i in range(nb):
    band = Image.GetRasterBand(i+1)
    print("band name", band.GetDescription())
    band_names.append(band.GetDescription().replace(",", "_").strip())

out_hdr = "image,row,lin" 
for i in range(nb):
    out_hdr += "," + band_names[i]
out_spec_f.write(out_hdr.encode())

if True:
    if True:
        plt.figure(figsize=(16, 16), dpi=300)
        plt.title('Spectra ') #  + args[1] + ' row: ' + args[2] + ' col ' + args[3])
        for p_i in range(len(pix_j)):
            count = 0
            cmd = ["gdallocationinfo",
                   img, # input image
                   str(pix_j[p_i]), # default: pixl number (0-indexed) aka row
                   str(lin_j[p_i])] # default: line number (0-indexed) aka col
            cmd = ' '.join(cmd)
            print("  \t" + cmd)
            lines = [x.strip() for x in os.popen(cmd).readlines()]

            for line in lines:
                print('\t' + line)
            if len(lines) != 2 * (1 + nb):
                err("unexpected result line count")

            w = lines[1].split()
            if w[0] != "Location:":
                err("unexpected field")
            pix_k, lin_k = w[1].strip('(').strip(')').split(',')
            if pix_k[-1] != 'P' or lin_k[-1] != 'L':
                err('unexpected data')

            pix_k, lin_k = int(pix_k[:-1]), int(lin_k[:-1])
            print(str(pix_k) + 'P ' + str(lin_k) + 'L')
            count += 1
            data = []
            for k in range(0, nb): # for each band
                bn = lines[2 * (1 + k)].strip(":").strip().split()
                if int(bn[1]) != k + 1:
                    err("expected: Band: " + str(k + 1) + "; found: " + lines[2 * (1 + k)])

                value = float(lines[3 + (2*k)].split()[1].strip())
                data.append(value)
            if False:
                print("\tdata", data)
            data_line = [img, pix_k, lin_k] + data
            data_line = ','.join([str(x) for x in data_line])
            out_spec_f.write(("\n" + data_line).encode())
            plt.plot(range(nb), data, label=label_j[p_i] + ' col=' + str(pix_j[p_i]) + ' row=' + str(lin_j[p_i]))
        plt.xticks(range(nb), band_names, rotation='vertical')
        # plt.tight_layout()
        plt.legend()
        plt.savefig(out_spec_fn[:-4] + '.png')

out_spec_f.close()
print("number of spectra extracted:", count)
