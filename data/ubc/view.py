''' 20231230 view.py e.g. Usage:
      python3 view.py G80223_20230513.tif
      python3 view.py G90292_20230514.tif
'''
import sys
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

tif_file_path = sys.argv[1] # "path/to/your/file.tif"
dataset = gdal.Open(tif_file_path)

def scale(rgb):
    mymin = np.nanmin(rgb) # np.nanmin(rgb))
    mymax = np.nanmax(rgb) # np.nanmax(rgb))
    print(mymin, mymax)

    rgb -= mymin
    rgb /= (mymax - mymin)

    rgb[rgb < 0.] = 0.  # clip
    rgb[rgb > 1.] = 1.
    return rgb

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


