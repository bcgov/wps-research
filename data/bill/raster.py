'''
Read raster data from .bin file (ENVI format)

Syntax for quick plot
  >> python3 raster.py file.bin
'''

from osgeo import gdal

from misc.general import htrim_3d

from misc.sen2 import (
    read_raster_timestamp,
    band_index
)

from plot_tools import plot

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

        return np.frombuffer(raw, dtype=np.float32).reshape((ysize, xsize))



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
            band_lst = 'all',
            info = False
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


        if info:

            #Print information of the selected bands
            print("---------------------")
            print("Your selected bands:")
            for b in band_lst:
                print(self.band_info_list[b-1])
            print("---------------------")
            print(f'Data Dimension: {data.shape}')


        return data
    


    def readBands_and_trim(
            self,
            band_lst='all',
            *,
            p = 1.,
            info = False
    ):
        
        '''
        Description
        -----------
        A pipeline style

        The same as read -> trim, but you cannot do anything in between, refers to Notes

        The effective matrix is saved as an attribute. Any other changes will override this matrix.

        
        Parameters
        ----------
        band_lst: In the meta data, each band will be ordered and can be accessed from index 1.

        p: percentage used for histogram trimming.

        info: to print out information of selected bands

        
        Returns
        -------
        A 3D matrix, each layer represents each band.


        Notes
        -----
        For now, this version only supports for 3 band list.

        Less flexible if data needs to be preprocessed

        Recommend using read_rgb -> see for changes -> trim

        For cropping, use with caution since it can returns matrix in unwanted shape.
        '''
        
        rgb = self.read_bands(
            band_lst=band_lst,
            info=info
        )

        return htrim_3d(
            rgb, p
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
    


    def is_polygon(
            self
    ):
        
        '''
        Description
        -----------
        Checks whether the loaded data is really a polygon.

        
        Must pass
        ---------
        
            1. Has only 1 band

            2. Is of boolean dtype
        '''
        from misc.general import is_boolean_matrix


        if (self._n_band > 1):

            print('This file has more than 1 band. Not a polygon')

            return False


        if ( not is_boolean_matrix(self.read_bands('all')) ):

            print('dtype of polygon is not boolean. Not a polygon.')

            return False
        
        
        return True
    


    def can_match_with(
            self,
            imagery
    ):
        '''
        Description
        -----------
        Checks if this imagery is applicable with another one.
        '''

        #Check shape

        if (self._xSize != imagery._xSize) or (self._ySize != imagery._ySize):

            print('Polygon cant be broadcast with Raster -> Mismatched shape')
            return False
        

        #Check projection

        if (self._proj != imagery._proj):

            print(f'{self._proj} doesn not match {imagery._proj}')
            return False
        

        #Check transform

        if not np.allclose(self._transform, imagery._transform, atol=0):

            print('Geotransform mismatch.')
            return False
        
        
        return True
    


    def band_name(
            self,
            band_index
    ):
        '''
        input band_index starts at 1.
        '''
        from misc.sen2 import band_name

        return band_name(
            band_info_list=self.band_info_list, 
            band_index=band_index - 1
        )




def minimum_nan_raster(
        filename_lst:list
):
    '''
    Description
    -----------
    Compares between the rasters to select the one with the least No Data in count.
    '''

    from misc.general import ignore_nan_2D

    min_nan_count = np.inf

    best_raster = None

    for f in filename_lst:

        raster = Raster(file_name=f)

        raster_data = raster.read_bands()

        X = raster_data.reshape(-1, raster._n_band)

        mask, _ = ignore_nan_2D(
            X = X,
            axis=1
        )

        nan_count = np.sum(~mask)

        if (nan_count < min_nan_count):

            min_count = nan_count

            best_raster = raster


    return best_raster, min_count




if __name__ == "__main__":

    #handling argv
    if len(sys.argv) < 2:
        print("Needs a raster file name")
        sys.exit(1)

    print('Reading Raster...')
    raster_filename = sys.argv[1]

    #load raster and read
    raster = Raster(file_name = raster_filename)

    if raster._n_band == 1:
        raster_dat = raster.read_bands('all').squeeze()

    else:
        raster_dat = raster.readBands_and_trim(band_lst=[1,2,3])

    if len(sys.argv) > 2:

        from misc.general import (
            extract_border,
            draw_border
        )

        print('Reading Polygon...')
        polygon_filename = sys.argv[2]

        print('Applying polygon onto raster...')
        polygon_dat = Raster(polygon_filename).read_bands(band_lst=[1])
        border = extract_border(
            polygon_dat.squeeze(),
            thickness=5
        )
        raster_dat = draw_border(raster_dat, border)

    #plot result
    plot(raster_dat)