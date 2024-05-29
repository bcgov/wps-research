''' 20231230 view.py e.g. Usage:
      python3 view.py G80223_20230513.tif
      python3 view.py G90292_20230514.tif
'''
import sys
import math
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt


def scale(X, use_histogram_trimming=True, use_clip=False, percent_trim_factor=None):
    # default: scale a band to [0, 1]
    mymin = np.nanmin(X) # np.nanmin(X))
    mymax = np.nanmax(X) # np.nanmax(X))
    X = (X-mymin) / (mymax - mymin)  # perform the linear transformation

    # use histogram trimming / turn it off to see what this step does!
    if use_histogram_trimming:
        values = X.ravel().tolist()
        values.sort()
        n_pct = 1. # percent for stretch value

        if percent_trim_factor is not None:
            n_pct = float(percent_trim_factor)
        frac = n_pct / 100.
        lower = int(math.floor(float(len(values))*frac))
        upper = int(math.floor(float(len(values))*(1. - frac)))
        mymin, mymax = values[lower], values[upper]
        X = (X-mymin) / (mymax - mymin)  # perform the linear transformation

    if use_clip:
        X[X < 0.] = float('nan') # 0.  # clip
        X[X > 1.] = float('nan') # 1.

    return X


def plot(dataset, use_histogram_trimming=True, transect_line_ix=None):
    if type(dataset) == str:
        dataset = gdal.Open(dataset)

    # image dimensions
    width = int(dataset.RasterXSize)
    height = int(dataset.RasterYSize)

    rgb = np.zeros((height, width, 3))

    # Read the data from the raster bands (assuming RGB bands are 1, 2, and 3)
    for i in range(3):
        rgb[:, :, i] = scale(dataset.GetRasterBand(i+1).ReadAsArray().reshape((height, width)), use_histogram_trimming)
    ''' A data cube indexed by row, column and band index (band index is in 1,2,3 rather: 0,1,2 from 0) 

    0,1,2 are not actually red, green blue. They are B12, B11, B9 from Sentinel-2:
        B12: 2190 nm
        B11: 1610 nm
        B9: 940 nm 
    which are in the short-wave infrared (SWIR)

    We chose the false-color encoding (R,G,B) = (B12, B11, B9) because fire looks orange/red/brown ish
    ''' 
   
    # Close the dataset
    dataset = None

    # Plot the RGB image using Matplotlib
    plt.figure(figsize=(7.5, 7.5))
    plt.title("R,G,B =(B12, B11, B9) for " + sys.argv[1])
    plt.imshow(rgb)
    if transect_line_ix != None:
        plt.axhline(y = transect_line_ix, color = 'black', linestyle = '--', linewidth = 4, alpha=.5) 
        # plt.hlines(transect_line_ix, color='black') 
    # plt.axis('off')  # Turn off axis labels
    # plt.tight_layout()
    print('figsize', plt.rcParams["figure.figsize"])
    plt.show()
    plt.rcParams["figure.figsize"] = (6.4, 4.8)


if __name__ == '__main__':
    tif_file_path = sys.argv[1] # "path/to/your/file.tif"
    dataset = gdal.Open(tif_file_path)

    # Check if the dataset was successfully opened
    if not dataset:
        print(f"Failed to open the TIF file: {tif_file_path}")
    else:  
        plot(dataset)
