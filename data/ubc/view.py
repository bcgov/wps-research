''' 20231230 view.py e.g. Usage:
      python3 view.py G80223_20230513.tif
      python3 view.py G90292_20230514.tif
'''
import sys
import math
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

tif_file_path = sys.argv[1] # "path/to/your/file.tif"
dataset = gdal.Open(tif_file_path)

def scale(X):
    # default: scale a band to [0, 1]  and then clip
    mymin = np.nanmin(X) # np.nanmin(X))
    mymax = np.nanmax(X) # np.nanmax(X))
    X = (X-mymin) / (mymax - mymin)  # perform the linear transformation

    X[X < 0.] = 0.  # clip
    X[X > 1.] = 1.

    # use histogram trimming / turn it off to see what this step does!
    if True:
        values = X.ravel().tolist()
        values.sort()
        n_pct = 1. # percent for stretch value
        frac = n_pct / 100.
        lower = int(math.floor(float(len(values))*frac))
        upper = int(math.floor(float(len(values))*(1. - frac)))
        mymin, mymax = values[lower], values[upper]
        X = (X-mymin) / (mymax - mymin)  # perform the linear transformation

    return X

# Check if the dataset was successfully opened
if not dataset:
    print(f"Failed to open the TIF file: {tif_file_path}")
else:
    # image dimensions
    width = int(dataset.RasterXSize)
    height = int(dataset.RasterYSize)

    rgb = np.zeros((height, width, 3))

    # Read the data from the raster bands (assuming RGB bands are 1, 2, and 3)
    rgb[:, :, 2] = scale(dataset.GetRasterBand(1).ReadAsArray().reshape((height, width)))
    rgb[:, :, 1] = scale(dataset.GetRasterBand(2).ReadAsArray().reshape((height, width)))
    rgb[:, :, 0] = scale(dataset.GetRasterBand(3).ReadAsArray().reshape((height, width)))

    print(rgb)
    
    # Close the dataset
    dataset = None

    # Plot the RGB image using Matplotlib
    plt.figure()
    plt.imshow(rgb)
    plt.title('RGB TIF File Visualization')
    plt.axis('off')  # Turn off axis labels
    plt.tight_layout()
    plt.show()


