# this script adapted from the one we worked on here:
#   https://github.com/franarama/satellite-clustering/blob/master/cluster.py
import sys
import math
import numpy as np
from misc import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from osgeo import gdal, gdal_array

# init GDAL
gdal.UseExceptions()
gdal.AllRegister()

if len(args) < 3:
    err("kmeans.py [input file: ENVI binary, BSQ, type 4] [# of clusters]")

K = 10

# parse command line arg
try:
    image = args[1]
    K = int(args[2])
except:
    err("failed to open input file: " + args[1])

# Read in raster image
img_ds = gdal.Open(image, gdal.GA_ReadOnly)

img = np.zeros((img_ds.RasterYSize,img_ds.RasterXSize, img_ds.RasterCount),
                gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))

for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

new_shape = (img.shape[0] * img.shape[1], img.shape[2])
X = img[:, :, :img.shape[2]].reshape(new_shape)

k_means = KMeans(n_clusters=K)
k_means.fit(X)

X_cluster = k_means.labels_
X_cluster = X_cluster.reshape(img[:, :, 0].shape)

fn = args[1] + '_kmeans.'
write_binary(X_cluster, fn + 'bin')

# write a header
samples, lines, bands = read_hdr(hdr_fn(args[1]))
write_hdr(fn + 'hdr', samples, lines, 1)  # class map, one band
