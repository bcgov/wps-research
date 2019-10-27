# use fastclust library https://pypi.org/project/fastcluster/
# for unsupervised learning on an image (next: object-based version)

# tested with python 2.7 on ubuntu
import os
import sys
from misc import *

def impt(lib):
    try:
        exec('import ' + lib)
    except:
        raw_input("pls. press ENTER to install " + lib + " (or ctrl-C to exit)" + 
                "\n\nnote: if you get any error, \n" +
                "pls. try to run fast_clust.py once more after.)\n")
        try:
            os.system('pip install ' + lib)
        except:
            err("please run application again")
            sys.exit(0)

impt('numpy'); import numpy as np
impt('fastcluster'); import fastcluster as fc
impt('matplotlib'); import matplotlib.pyplot as plt 
impt('gdal'); from osgeo import gdal, gdal_array
impt('scipy'); from scipy.cluster.hierarchy import dendrogram, linkage

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()
n_clusters = None 

try: image, n_clusters = sys.argv[1], int(sys.argv[2])
except: err("Usage: python fastcluster.py [image file name .bin] [n clusters desired]")

# Read in raster image
img_ds = gdal.Open(image, gdal.GA_ReadOnly)

# allocate memory to reshape image
img = np.zeros((img_ds.RasterYSize,  # number of rows
                img_ds.RasterXSize,  # number of cols
                img_ds.RasterCount),  # number of bands
                gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType)) # data type code
#print img.shape # warning: that assumed that the raster bands were all the same type (should be true)

# reshape the image band by band
for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

# reshape image again to match expected format for scikit-learn
new_shape = (img.shape[0] * img.shape[1], img.shape[2])
X = img[:, :, :img.shape[2]].reshape(new_shape)

# use fastcluster.linkage instead of scipy.cluster.hierarchy.linkage
print "calculating linkage.."
Z = fc.linkage(X, 'ward')

print "calculating dendrogram.."
fig = plt.figure(figsize=(10, 10)) # 25, 10
plt.title('hierarchical clustering dendrogram')
rotate = False

plt.ylabel('distance' if (not rotate) else 'index')
plt.xlabel('index' if (not rotate) else 'distance')
dn = dendrogram(Z,
        truncate_mode='lastp',
        p = n_clusters,
        leaf_rotation = 0. if rotate else 90.,
        show_contracted=True,
        orientation='right' if rotate else 'top',
        distance_sort='descending',
        show_leaf_counts=True)

print "saving figure.."
plt.savefig(image + "_fastcluster.png")

print "extracting labels.."
from scipy.cluster.hierarchy import fcluster
labels = fcluster(Z, n_clusters, criterion='maxclust') ##criterion='distance')

labels=labels.reshape(img[:, :, 0].shape)
plt.figure(figsize=(20, 20))
print "min, max", np.min(labels), np.max(labels)
plt.imshow(labels) #, cmap ='jet')
#plt.colorbar()
plt.tight_layout() #plt.show()
plot_file = image + "_labels.png"

print "plot written to file: " + plot_file
plt.savefig(plot_file, orientation='portrait')

samples, lines, bands = read_hdr(hdr_fn(args[1]))
write_binary(labels, args[1] + "_fastclust.bin")
write_hdr(args[1] + "_fastclust.hdr", samples, lines, 1)

