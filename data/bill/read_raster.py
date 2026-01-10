'''
Read raster data from .bin file (need .hdr ready as well)

python3 read_raster.py file.bin
'''

from osgeo import gdal

from misc.general import (
    htrim_3d, 
    crop_no_data
)

from misc.sen2 import (
    read_raster_timestamp,
    band_index
)

from plot_tools import (
    plot
)

import numpy as np

import sys


#Handling exception
gdal.UseExceptions()


class Raster:

    def __init__(
            self, 
            file_name
        ):

        '''
        At the moment a file_name is passed in, gdal will extract all of its information.
        '''

        self.file_name = file_name

        #Extract dataset from raw raster file

        self.__read_data()

    
    def __read_data(
            self
    ):
        '''
        Read and extract
        Data + META DATA
        '''
        fname = self.file_name

        #Extract Data

        ds = gdal.Open(fname, gdal.GA_ReadOnly)

        self._dataset = ds

        if self._dataset is None:
            raise RuntimeError(f"Could not open {fname}")
        

        #Extract Meta Data
        
        self.acquisition_timestamp = read_raster_timestamp(fname) #Acquisition data and time (UTC)

        self._xSize, self._ySize = ds.RasterXSize, ds.RasterYSize
        self._count = ds.RasterCount
        self._proj = ds.GetProjection()
        self._transform = ds.GetGeoTransform()


        #Extract band information

        self._n_band = ds.RasterCount
        self.band_info_list = [
            ds.GetRasterBand(i).GetDescription() for i in range(1, 
                                                                ds.RasterCount + 1)
        ]



    def __read_single_band(
            self,
            band
        ):

        '''
        Helper function

        Returns a 2D array of the chosen band

        Parameters
        ----------
        band: The specific band to cut from the data.

        Returns
        -------
        A 2D array of the chosen band (a single layer)
        '''

        band = self._dataset.GetRasterBand(band)

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



    def read_bands(
            self,
            band_tup = (1,2,3)
    ):
        '''
        Read each band and arrange everything in a 3D matrix.

        Parameters
        ----------
        band_tup: In the meta data, each band will be ordered and can be accessed from index 1.
        

        Returns
        -------
        A 3D matrix, each layer represents a band.


        Note
        ----
        band_tup can be any of any length. Will fix.

        Generally if use just the first 3 bands, there is no problem.
        '''

        #Handle band out of bound here

        from exceptions.sen2_exception import Out_Of_Bound_Band_Index


        if (min(band_tup) < 1 or max(band_tup) > self._n_band):

            raise Out_Of_Bound_Band_Index(f'Some band index is not in the data.')


        stack = np.dstack([
            self.__read_single_band(b) for b in band_tup
        ])

        return stack
    


    def readBands_and_trim(
            self,
            *,
            p = 1.,
            crop = False
    ):
        
        '''A pipeline

        The same as read -> trim, but you cannot do anything in between, refers to Notes

        The effective matrix is saved as an attribute. Any other changes will override this matrix.

        
        Parameters
        ----------
        p: percentage used for histogram trimming.

        crop: boolean to remove rows and columns that may contain mostly repeated values (no data there).

        
        Returns
        -------
        A 3D matrix, each layer represents each band.


        Notes
        -----
        Less flexible if data needs to be preprocessed

        Recommend using read_rgb -> see for changes -> trim

        For cropping, use with caution since it can returns matrix in unwanted shape.
        '''
        
        rgb = self.read_bands()

        if crop: rgb = crop_no_data(rgb)

        return htrim_3d(
            rgb, 
            p
        )
    


    def get_band(
            self,
            band
    ):
        '''
        Get the layer using the band name (integer).

        Parameters
        ----------
        band: The wave to get


        Returns
        -------
        Band array (2D)
        '''

        band_id = band_index(self.band_info_list, band)

        band = self.__read_single_band(band_id + 1) #Because gdal read band starts at 1

        return band




if __name__ == "__main__":

    #handling argv
    if len(sys.argv) < 2:
        print("Needs a file name")
        sys.exit(1)

    filename = sys.argv[1]

    #load raster and read
    raster = Raster(file_name=filename)

    X = raster.readBands_and_trim(crop=True)

    #Plot title
    title = raster.acquisition_timestamp
    
    if len(sys.argv) > 2: title = sys.argv[2]

    #plot result
    plot(
        X,
        title = title
    )