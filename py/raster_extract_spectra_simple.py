# this script extracts one spectrum only from a raster: from at a row, col point location
# considered to be the "trusted" spectra extraction
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
if len(args) < 4:
    err("python3 extract_spectra.py [input image name] [row] [col]")

img = args[1] # input image
if not os.path.exists(img):
    err('file not found: ' + img)

pix_j = int(args[3]) # col 0-idx
lin_j = int(args[2]) # row 0-idx

out_spec_fn = img + "_spectrum_row_col_" + args[2] + "_" + args[3] + ".csv"
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
        if True:
            count = 0
            cmd = ["gdallocationinfo",
                   img, # input image
                   str(pix_j), # default: pixl number (0-indexed) aka row
                   str(lin_j)] # default: line number (0-indexed) aka col
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
        
            plt.figure()
            plt.title('Spectra ' + args[1] + ' row: ' + args[2] + ' col ' + args[3])
            plt.plot(range(nb), data)
            plt.xticks(range(nb), band_names, rotation='vertical')
            plt.tight_layout()
            plt.savefig(out_spec_fn[:-4] + '.png')

out_spec_f.close()
print("number of spectra extracted:", count)
