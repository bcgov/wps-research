''' 20260107  usage:
     python3 view.py G80223_20230513.bin # your image filename goes here.. '''
#!/usr/bin/env python3
import sys
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt


def scale(X, p=1.0):
    """Percentile-based contrast stretch.
    p = lower/upper percentile to clip (e.g. 1.0 -> 1% / 99%)"""
    X = X.astype(np.float32)
    lo, hi = np.percentile(X, [p, 100 - p])
    return (X - lo) / (hi - lo)


def plot(data, title=None):
    """
    Plot Sentinel-2 SWIR false color composite:
    R,G,B = B12, B11, B9
    """

    # Read bands (1-based indexing in GDAL)
    rgb = np.dstack([scale(data.GetRasterBand(1).ReadAsArray()),  # B12
                     scale(data.GetRasterBand(2).ReadAsArray()),  # B11
                     scale(data.GetRasterBand(3).ReadAsArray())])  # B9
    plt.figure() 
    plt.imshow(rgb)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: plot_swir.py <raster>")
        sys.exit(1)

    filename = sys.argv[1]
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Could not open {filename}")

    title = f"{filename} False color R,G,B = (B12, B11, B9)"
    plot(dataset, title)


if __name__ == "__main__":
    main()
