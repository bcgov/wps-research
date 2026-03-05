from osgeo import gdal

import numpy as np

from pathlib import Path

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

        self.NODATA_VALUE = np.nan

        self.__read_data()

    
    def __read_data(
            self
    ):
        '''
        Read and extract

        Data + META DATA
        '''

        #Extract Data
        ds = gdal.Open(self.file_name, gdal.GA_ReadOnly)

        self._dataset = ds

        if self._dataset is None:
            raise RuntimeError(f"Could not open {self.file_name}")
        
        #Information from filename
        filename = Path(self.file_name).name  # BUG FIX: was Path(filename)

        file_fields = filename.split('_')

        # Gracefully handle filenames that don't follow the expected convention
        try:
            self.level = file_fields[1][4]
            self.acquisition_time = file_fields[2]
            self.grid = file_fields[5]
        except (IndexError, KeyError):
            self.level = None
            self.acquisition_time = None
            self.grid = None

        #Extract geoinfo
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

        #There is case where only 1 band is in the data, and no info is given.
        if (self._n_band == 1) and (self.band_info_list[0] == ''):

            self.band_info_list[0] = 'gray_scale, unknown band'



    def __read_single_band(
            self,
            band: int
        ):

        '''
        Description 
        -----------
        This function is used internally to read a single band. 

        
        Parameters
        ----------
        band: A specific band index.

        
        Returns
        -------
        A 2D array of the chosen band (a single layer)
        '''

        if not isinstance(band, int):

            raise TypeError("band must be an integer.")

        if ( band < 1 or band > self._n_band ): 
            '''
            My convention is that band number starts at 1.

            The max is number of bands.
            '''

            raise ValueError(
                f"band must be between 1 and {self._n_band}, got {band}"
            )

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

        arr = np.frombuffer(raw, dtype=np.float32).reshape((ysize, xsize))

        return arr



    def __ite_read(
            self,
            band_lst
    ):
        '''
        Description
        -----------
        Iteratively reads and saves all band in a stack using __read_single_band(band)
        '''

        return np.dstack([
                self.__read_single_band(b) for b in band_lst
        ])



    def read_bands(
            self,
            band_lst = 'all'
    ):
        '''
        Description
        -----------
        Read each band and arrange everything in a 3D matrix.


        Parameters
        ----------
        band_lst: In the meta data, each band will be ordered and can be accessed from index 1. Default is 'all' band(s).

        info: to print out information of selected bands
        

        Returns
        -------
        A 3D matrix, each layer represents a band.

        Prints information of the selected bands


        Note
        ----
        band_tup can be any of any length. Will fix.

        Generally if use just the first 3 bands, there is no problem.
        '''

        #Handle band out of bound here


        if band_lst == 'all':

            data = self.__ite_read(band_lst = list(range(1, self._n_band + 1)))


        else:

            if (min(band_lst) < 1 or max(band_lst) > self._n_band):

                raise IndexError(f'Some band index is not in the data.')


            data = self.__ite_read(band_lst = band_lst)


        return data