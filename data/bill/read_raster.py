'''
Read raster data from .bin file (need .hdr ready as well)

Update: Will add an auto trim, for visualization

python3 read_raster.py file.bin
'''

from osgeo import gdal
gdal.UseExceptions()

from misc import (
    htrim_rgb, 
    crop
)

import numpy as np

import sys


class Raster:

    def __init__(
            self, 
            file_name
        ):

        self.file_name = file_name

        #Extract dataset from raw raster file
        self._dataset = gdal.Open(file_name, gdal.GA_ReadOnly)

        if self._dataset is None:
            raise RuntimeError(f"Could not open {file_name}")
        

    def _read_single_band(
            self, 
            data, 
            band
        ):
        '''
        Helper function
        Returns a 2D array of the chosen band
        '''
        band = data.GetRasterBand(band)

        xsize = band.XSize
        ysize = band.YSize

        if not hasattr(self, "xsize"):
            self.xsize = xsize
            self.ysize = ysize

        raw = band.ReadRaster(
            0, 0, xsize, ysize,
            buf_type=gdal.GDT_Float32
        ) 

        return np.frombuffer(raw, dtype=np.float32).reshape((ysize, xsize))


    def read(
            self,
            band_tup = (1,2,3)
    ):
        '''
        Read values of bands
        returns multi channel matrix
        '''

        return np.dstack([
            self._read_single_band(self._dataset, b) for b in band_tup
        ])
    


    def read_trim(
            self,
            *,
            p = 1.,
            _crop = True
    ):
        
        '''A pipeline
        The same as read -> trim
        Less flexible if data needs to be preprocesses
        Recommend using read_rgb -> see for changes -> trim
        '''
        
        rgb = self.read()

        if _crop: rgb = crop(rgb)

        return htrim_rgb(
            rgb, p
        )
    

    def plot(
            self, 
            X,
            *,
            title = None,
            figsize = (10, 8)
    ):
        
        '''
        Will use mainly for linux quick plot

        python3 read_raster.py file.bin [p - optional]

        Where: p is for htrim, default is 1, or 1%
        '''

        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize) 
        plt.imshow(X)
        plt.title(title)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    #handling argv
    if len(sys.argv) < 2:
        print("Needs a file name")
        sys.exit(1)

    filename = sys.argv[1]

    #load raster and read
    raster = Raster(file_name=filename)

    X = raster.read_trim()

    #plot result
    raster.plot(
        X
    )