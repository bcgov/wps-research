# Something wrong with this script? Don't use for now..

# this script extracts spectra (from a raster) at point locations in a shapefile
# example:
#  python3 extract_spectra.py FTL_test1.shp S2A_MSIL2A_20190908T195941_N0213_R128_T09VUE_20190908T233509_RGB.bin 20 10

# extracts on grid pattern with specified radius..
#.. not sure if GeoTiffs store band names, we read in from ENVI format Float32 file
import os
import sys
import json
import struct
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
args = sys.argv

def err(m):
    print("Error: " + str(m)); sys.exit(1)

if len(args) < 5:
    err("python3 extract_spectra.py [input shapefile name] [input image name] [round extraction window radius (m)] [ image res(m) ]")

shp, img = args[1], args[2] # input shapefile, image
if not os.path.exists(shp): err('file not found: ' + shp)
if not os.path.exists(img): err('file not found: ' + img)

out_spec_fn = img + "_spectra.csv"
out_spec_f = open(out_spec_fn, "wb")

res, rad = float(args[3]), float(args[4])  # assert numbers
cmd = ("python3 " +
       "/home/" + os.popen("whoami").read().strip() + "/GitHub/wps-research/py/raster_extract_window_offset.py " + args[3] + " " + args[4])

print(cmd)
a = os.system(cmd)

x_off = [int(i) for i in open(".x_off").read().strip().split(",")]
y_off = [int(i) for i in open(".y_off").read().strip().split(",")]

# Open image
Image = gdal.Open(img, gdal.GA_ReadOnly)
nc, nr, nb = Image.RasterXSize, Image.RasterYSize, Image.RasterCount # rows, cols, bands

band_names = []
for i in range(nb):
    band = Image.GetRasterBand(i+1)
    print("band name", band.GetDescription()) # andName())
    band_names.append(band.GetDescription().replace(",", "_").strip())

out_hdr = "feature_id,ctr_lat,ctr_lon,image,row,lin,xoff,yoff" 
for i in range(nb):
    out_hdr += "," + band_names[i]
out_spec_f.write(out_hdr.encode())

print("projection", Image.GetProjection)
proj = osr.SpatialReference(wkt=Image.GetProjection())
EPSG = proj.GetAttrValue('AUTHORITY', 1)
EPSG = int(EPSG)
print("Image EPSG", EPSG)

# Open Shapefile
Shapefile = ogr.Open(shp)
layer = Shapefile.GetLayer()
layerDefinition, feature_count = layer.GetLayerDefn(), layer.GetFeatureCount()
print("Shapefile spatialref:", layer.GetSpatialRef())

def records(layer):
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        yield json.loads(feature.ExportToJson())
print("feature count: " + str(feature_count))
features = records(layer)

feature_names, feature_ids, coordinates = [], [], []
for f in features: # print(f.keys())
    feature_id = f['id']
    feature_ids.append(feature_id) # print("feature properties.keys()", f['properties'].keys())
    
    feature_name = ''
    try:
        feature_name = f['properties']['Name']
    except Exception:
        pass # feature name not available
    feature_names.append(feature_name)
    
    fgt = f['geometry']['type']
    if fgt != 'Point':
        err('Point geometry expected. Found geometry type: ' + str(fgt))
    coordinates.append(f['geometry']['coordinates'])
    #    print("geom", f) # ['geometry'])

count = 0 # extract spectra
for i in range(feature_count):
    print(feature_ids[i], coordinates[i])
    
    # not efficient for "many" points
    cmd = ["gdallocationinfo",
           img, # input image
           '-wgs84', # specify lat long input
           str(coordinates[i][0]), # lat
           str(coordinates[i][1])] # long
    cmd = ' '.join(cmd)
    print(cmd)
    lines = [x.strip() for x in os.popen(cmd).readlines()]
    

    if len(lines) >= 2 * (1 + nb):
        w = lines[1].split()
        if w[0] != "Location:":
            err("unexpected field")
        pix_i, lin_i = w[1].strip('(').strip(')').split(',')
        if pix_i[-1] != 'P' or lin_i[-1] != 'L':
            err('unexpected data')

        pix_i, lin_i = int(pix_i[:-1]), int(lin_i[:-1])
        print(str(pix_i) + 'P ' + str( lin_i) + 'L')
        count += 1
        data = []
        for j in range(0, nb): # for each band
            bn = lines[2 * (1 + j)].strip(":").strip().split()
            if int(bn[1]) != j + 1:
                err("expected: Band: " + str(j + 1) + "; found: " + lines[2 * (1 + j)])

            value = float(lines[3 + (2*j)].split()[1].strip())
            data.append(value)
        if False:
            print("centre_data", data)
        # feature,ctr_lat,ctr_lon,row,lin,xoff,yoff,b0,b1,b2,b3
        data_line = [feature_ids[i], coordinates[i][0], coordinates[i][1],
                     img, pix_i, lin_i, 0, 0] + data
        data_line = ','.join([str(x) for x in data_line])
        out_spec_f.write(("\n" + data_line).encode())     
        
        for j in range(len(x_off)): # for each of the non-centre window points
            xo, yo = x_off[j], y_off[j]
            pix_j, lin_j = pix_i + xo, lin_i + yo
            print(pix_j, lin_j, xo, yo, pix_i, lin_i)
            
            cmd = ["gdallocationinfo",
                   img, # input image
                   str(pix_j), # default: pixl number (0-indexed) aka row
                   str(lin_j)] # default: line number (0-indexed) aka col
            cmd = ' '.join(cmd)
            print("  \t" + cmd)
            lines = [x.strip() for x in os.popen(cmd).readlines()]
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
            data_line = [feature_ids[i], coordinates[i][0], coordinates[i][1], img, pix_k, lin_k, xo, yo] + data
            data_line = ','.join([str(x) for x in data_line])
            out_spec_f.write(("\n" + data_line).encode())

out_spec_f.close()
print("number of spectra extracted:", count)
