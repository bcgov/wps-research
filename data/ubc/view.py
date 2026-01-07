''' 
20260107  usage:
     python3 view.py G80223_20230513.bin # your image filename goes here.. 

20231230 view.py e.g. Usage:
      python3 view.py G80223_20230513.tif
      python3 view.py G90292_20230514.tif
'''
import sys
import math
import numpy as np
from osgeo import gdal  # library for reading .bin file
import matplotlib.pyplot as plt


def scale(X, P_percent = 1.):  # 
    x = X.ravel().tolist(); x.sort()
    N, p = float(len(x)), P_percent / 100.
    x0 = x[int(math.floor(p * N))]
    x1 = x[int(math.floor((1 - p) * N))]
    return  (X- x0) / ( x1 - x0 ) 

''' A data cube indexed by row, column and band index (band index is in 1,2,3 rather: 0,1,2 from 0) 

    0,1,2 are not actually red, green blue. They are B12, B11, B9 from Sentinel-2:
        B12: 2190 nm
        B11: 1610 nm
        B9: 940 nm 
    which are in the short-wave infrared (SWIR)

    We chose the false-color encoding (R,G,B) = (B12, B11, B9) because fire looks orange/red/brown ish
'''
def plot(dataset):
    # image dimensions
    width = int(dataset.RasterXSize)  # ncol
    height = int(dataset.RasterYSize) # nrow
    rgb = np.zeros((height, width, 3))

    # Read the data from the raster bands (assuming RGB bands are 1, 2, and 3)
    rgb[:, :, 0] = scale(dataset.GetRasterBand(1).ReadAsArray().reshape((height, width)))
    rgb[:, :, 1] = scale(dataset.GetRasterBand(2).ReadAsArray().reshape((height, width)))
    rgb[:, :, 2] = scale(dataset.GetRasterBand(3).ReadAsArray().reshape((height, width)))
   
    # Plot the RGB image using Matplotlib
    plt.figure()
    plt.imshow(rgb)
    plt.title(sys.argv[1] + " with encoding R,G,B =(B12, B11, B9)")
    plt.axis('off')  # Turn off axis labels
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    tif_file_path = sys.argv[1] # "path/to/your/file.tif"
    dataset = gdal.Open(tif_file_path)

    # Check if the dataset was successfully opened
    if not dataset:
        print(f"Failed to open the TIF file: {tif_file_path}")
    else:  
        plot(dataset)
    dataset = None # can probably delete this
