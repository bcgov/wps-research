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

linkages = {}
p_mns = {}
Xs = {}

for x in range(0, ntile_x):
    x_start = x * tile_size
    x_end = min(x_start + tile_size - 1, ncol - 1) 
    x_start = max(0, x_start - border_size) # add border
    x_end = min(ncol - 1, x_end + border_size)
    x_size = x_end - x_start + 1 #  calculate size

    for y in range(0, ntile_y):
        y_start = y * tile_size
        y_end = min(y_start + tile_size - 1, nrow - 1)
        y_start = max(0, y_start - border_size)  # add border
        y_end = min(nrow - 1, y_end + border_size)
        y_size = y_end - y_start + 1 # calculate size
        
        p = str(x) + "_" + str(y)
        pfx = image + "_" + p

        # plot image tile
        plt.figure()
        X = img[int(x_start): int(x_end), int(y_start): int(y_end),:]
        plt.imshow(twop_str(X))
        plt.tight_layout()
        pfx = image + "_" + str(x) + "_" + str(y)
        f_n = pfx + ".png"
        print "+w", f_n
        plt.savefig(f_n)
    
        Xs[p] = X
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


# threshold distance
d_t =  min(p_mns.values())

for x in range(0, ntile_x):
    x_start = x * tile_size
    x_end = min(x_start + tile_size - 1, ncol - 1)
    x_start = max(0, x_start - border_size) # add border
    x_end = min(ncol - 1, x_end + border_size)
    x_size = x_end - x_start + 1 #  calculate size

    for y in range(0, ntile_y):
        y_start = y * tile_size
        y_end = min(y_start + tile_size - 1, nrow - 1)
        y_start = max(0, y_start - border_size)  # add border
        y_end = min(nrow - 1, y_end + border_size)
        y_size = y_end - y_start + 1 # calculate size

        p = str(x) + "_" + str(y)
        pfx = image + "_" + p

        Z = linkages[p]
        X = Xs[p]

        labels = fcluster(Z, d_t, criterion = 'distance') #criterion='maxclust') ##criterion='distance')
        labels=labels.reshape(X.shape[0], X.shape[1]) #X[:, :].shape)
        # plt.figure(figsize=(20, 20))
        print "min, max", np.min(labels), np.max(labels)
        plt.imshow(labels)
        plt.tight_layout() #plt.show()
        plot_file = pfx + "_labels.png"

        print "plot written to file: " + plot_file
        plt.savefig(plot_file, orientation='portrait')

        write_binary(labels, pfx + ".bin")
        write_hdr(pfx + ".hdr", X.shape[0], X.shape[1], 1) # samples, lines, 1)



# isn't it a waste to calculate the full linkage, on something that's not merged all the way?

