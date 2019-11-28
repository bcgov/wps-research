# use fastclust library https://pypi.org/project/fastcluster/ for unsupervised learning on an image (next: object-based version). Tested with python 2.7 on ubuntu
# this adapts our initial work here: https://github.com/franarama/satellite-clustering/blob/hierarchical-clustering/cluster.py

# e.g.
# python py/fast_cluster_tiling.py  data/fran/mS2.bin 10 50 10

from misc import *

def impt(lib):
    try: exec('import ' + lib)
    except:
        raw_input("pls. press ENTER to install " + lib + " (or ctrl-C to exit)" + 
                "\n\nnote: if you get any error, \n" +
                "pls. try to run fast_clust.py once more after.)\n")
        try: os.system('pip install ' + lib)
        except: err("please run application again")

impt('numpy'); import numpy as np
impt('fastcluster'); import fastcluster as fc
impt('matplotlib'); import matplotlib.pyplot as plt 
impt('gdal'); from osgeo import gdal, gdal_array
impt('scipy'); from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()
n_clusters, tile_size, border_size = None, None, None

try: image, n_clusters, tile_size, border_size = args[1], int(args[2]), int(args[3]), int(args[4])
except: err("Usage: python fastcluster.py [image file name .bin] [n clusters desired]" +
            "[square tiles: number of pixels n (tile size: n x n)] " +
            "[border size around tile: number of pixels]")

# Read in raster image
img_ds = gdal.Open(image, gdal.GA_ReadOnly)

# allocate memory to reshape image
img = np.zeros((img_ds.RasterYSize,  # number of rows
                img_ds.RasterXSize,  # number of cols
                img_ds.RasterCount),  # number of bands
                gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
                # data type code #. warning: raster bands assumed all same type (should be )

# reshape the image band by band (NRow, NCol, NBand)
for b in range(img.shape[2]): img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

ncol, nrow, nband = img.shape
print "img.shape", nrow, ncol, nband

rx, ry = ncol / tile_size, nrow / tile_size
ntile_x = rx if ncol % tile_size == 0 else rx + 1
ntile_y = ry if nrow % tile_size == 0 else ry + 1

print "ntilex", ntile_x, "ntiley", ntile_y

# outputs from first step:
p_mns = {} # histogram distance thresholds, per tile
extent = {} # tile extents, per tile
X_shape = {} # data shape, per tile
linkages = {}

for x in range(0, ntile_x):
    x_start = x * tile_size
    x_end = min(x_start + tile_size, ncol)  # x_start + tile_size - 1, ncol - 1
    x_start = max(0, x_start - border_size) # add border
    x_end = min(ncol, x_end + border_size) # ncol - 1 
    x_size = x_end - x_start + 1 #  calculate size

    for y in range(0, ntile_y):
        y_start = y * tile_size
        y_end = min(y_start + tile_size, nrow)  # y_start + sile_size - 1, nrow - 1
        y_start = max(0, y_start - border_size)  # add border
        y_end = min(nrow, y_end + border_size) # nrow - 1
        y_size = y_end - y_start + 1 # calculate size
        
        # tile index
        p = str(x) + "_" + str(y)
        pfx = image + "_" + p

        # extract tile
        extent[p] = [x_start, x_end, y_start, y_end]
        X = img[int(x_start): int(x_end), int(y_start): int(y_end),:]
        
        # show tile
        plt.figure()
        plt.imshow(twop_str(X))
        f_n = pfx + ".png"
        print "+w", f_n
        plt.tight_layout()
        plt.savefig(f_n)
    
        print "calculating linkage.."
        X_r = X.reshape(X.shape[0] * X.shape[1], nband)

        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
        # average uses the average of the distances of each observation of the two sets.
        Z = fc.linkage(X_r, 'average') # https://en.wikipedia.org/wiki/UPGMA
        linkages[p] = Z

        values = Z.reshape(np.prod(Z.shape)).tolist()
        values.sort()
        q = 0.2
        pmn = min(values)
        p_mn = values[int(math.floor(q * len(values)))]
        p_mx = values[int(math.floor((1. - q) * len(values)))]
        pmx = max(values)
        print pmn, p_mn, p_mx, pmx
        p_mns[p] = p_mn


# threshold distance (only needed this, could run linkage again if too big to store)
d_t =  min(p_mns.values())
labels = {}

for x in range(0, ntile_x):
    for y in range(0, ntile_y):
        p = str(x) + "_" + str(y)
        pfx = image + "_" + p

        #Z, X = linkages[p], Xs[p] # could run again if too much memory to store
        Z = linkages[p]
        labels[p] = fcluster(Z, d_t, criterion = 'distance')
        labels[p] = labels[p].reshape(extent[p][1] - extent[p][0], extent[p][3] - extent[p][2])

        print "min, max", np.min(labels), np.max(labels)
        plt.figure()
        plt.imshow(labels[p])
        plt.tight_layout()
        plot_file = pfx + "_labels.png"

        print "plot written to file: " + plot_file
        plt.savefig(plot_file, orientation='portrait')

        print "+w", pfx + ".bin"
        labels[p] = labels[p].reshape(extent[p][3] - extent[p][2], extent[p][1] - extent[p][0], 1) #         extent[p][1] - extent[p][0], extent[p][3] - extent[p][2], 1)
        write_binary(labels[p], pfx + ".bin")
        write_hdr(pfx + ".hdr", extent[p][3] - extent[p][2], extent[p][1] - extent[p][0], 1) #extent[p][1] - extent[p][0], extent[p][3] - extent[p][2], 1) # samples, lines, 1)

out_shape = (img_ds.RasterYSize, img_ds.RasterXSize)
result = np.zeros(out_shape)

next_label = 0
for x in range(0, ntile_x):
    for y in range(0, ntile_y):
        p = str(x) + "_" + str(y)
        label = labels[p].astype(float)
        label = label.reshape(np.prod(label.shape)).tolist()
        min_label, max_label, n_label = min(label), max(label), int(max(label) - min(label) + 1)
        print "min_label", min_label, "max_label", max_label, "n_label", n_label, "start_label", next_label, "end_label", next_label + n_label - 1
        
        x_start, x_end, y_start, y_end = extent[p]
        print extent[p]
        labels[p] = labels[p].reshape(extent[p][1] - extent[p][0], extent[p][3] - extent[p][2])
        result[int(x_start): int(x_end), int(y_start): int(y_end)] = labels[p] + next_label 
        next_label += n_label

print "number of segments: ", next_label
plt.figure()
plt.imshow(result)
print "result.size", result.shape
print "img.shape", nrow, ncol, nband

plt.title("result")
plt.tight_layout()
#plt.xlim([0, img_ds.RasterXSize -1])
plt.savefig(image + "_result.png")

print  "**", img_ds.RasterYSize, img_ds.RasterXSize
result = result.astype(np.float32)
result = result.reshape(img_ds.RasterXSize, img_ds.RasterYSize, 1)
write_binary(result, image + "_label.bin")
write_hdr(image +"_label.hdr", img_ds.RasterXSize, img_ds.RasterYSize, 1)

#write_binary(
# isn't it a waste to calculate the full linkage, on something that's not merged all the way?

# pull the other stuff out into a function..


# 1) combine all the labels into one (with the appropriate offsets)

# 2) do h_clustering on the new class map



# IF NOT CALCULATING DENDROGRAM, DEFINITELY DON'T NEED TO SAVE DMAT
