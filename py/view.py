''' 20231230 view.py e.g. Usage:
      python3 view.py G80223_20230513.tif
      python3 view.py G90292_20230514.tif
'''
import sys
import math
import argparse
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


def plot(dataset, use_histogram_trimming=True, transect_line_ix=None, use_proportional=False):
    if type(dataset) == str:
        dataset = gdal.Open(dataset)
    band_names = [dataset.GetRasterBand(i).GetDescription() for i in range(1, dataset.RasterCount + 1)]

    # image dimensions
    width = int(dataset.RasterXSize)
    height = int(dataset.RasterYSize)

    rgb = np.zeros((height, width, 3))

    # Read the data from the raster bands (assuming RGB bands are 1, 2, and 3)
    if not use_proportional:
        for i in range(3):
            rgb[:, :, i] = scale(dataset.GetRasterBand(i+1).ReadAsArray().reshape((height, width)), use_histogram_trimming)
    else:
        for i in range(3):
            rgb[:, :, i] = dataset.GetRasterBand(i+1).ReadAsArray().reshape((height, width))
        intensity = np.zeros((height, width))

        if True:
            for i in range(height):
                for j in range(width):
                    intensity[i,j] = np.nanmax(rgb[i,j,:])
            
        intensities =  list(intensity.ravel())
        intensities.sort()
    
        n_pct = math.floor(0.01 * 1. * width * height)
        print(n_pct, width*height, len(intensities))
        my_min, my_max = intensities[n_pct], intensities[width * height - n_pct - 1]
        rgb = rgb - my_min
        rgb /= (my_max - my_min)

    # Close the dataset
    dataset = None

    # Plot the RGB image using Matplotlib
    plt.figure(figsize=(7.5, 7.5))
    plt.title("R,G,B =(" + ','.join(band_names) + ") for " + sys.argv[1])
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
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="input raster filename")
    parser.add_argument("-p", "--proportional_scaling", action="count", default=0, help="use proportional scaling, instead of nonproportional scaling")
    parser.add_argument("-n", "--no_histogram_stretching", action="count", default=0, help="set to 1 to turn off histogram stretching: min/max only")
    args = parser.parse_args()

    use_proportional = args.proportional_scaling  != 0
    histogram_stretching = args.no_histogram_stretching == 0

    tif_file_path = args.filename # sys.argv[1] # "path/to/your/file.tif"
    dataset = gdal.Open(tif_file_path)

    # Check if the dataset was successfully opened
    if not dataset:
        print(f"Failed to open the TIF file: {tif_file_path}")
    else:  
        plot(dataset, histogram_stretching, None, use_proportional)
