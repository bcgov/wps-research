import numpy as np
import rasterio
from rasterio.transform import from_origin
from osgeo import gdal
from misc import extract_date

def write_matrix_to_tif(matrix, data_file, output_tif):
    """
    Write a 2D matrix to a TIFF file using location data from a GDAL-readable data file.

    Parameters:
    - matrix (np.ndarray): 2D numpy array of values.
    - data_file (str): Path to the data file corresponding to the header file.
    - output_tif (str): Path to the output TIFF file.
    """
    print('+w', output_tif)

    def read_metadata_from_gdal(data_file):
        """
        Extract metadata using GDAL.
        """
        dataset = gdal.Open(data_file)
        if not dataset:
            raise FileNotFoundError(f"Unable to open data file: {data_file}")
        
        # Get geotransform and projection
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

        # Extract metadata
        if geotransform:
            ulx = geotransform[0]
            uly = geotransform[3]
            pixel_size_x = geotransform[1]
            pixel_size_y = geotransform[5]  # Usually negative if Y increases downward
        else:
            raise ValueError("Geotransform information is missing from the data file")

        # Dimensions
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize

        return {
            'ulx': ulx,
            'uly': uly,
            'pixel_size_x': pixel_size_x,
            'pixel_size_y': pixel_size_y,
            'rows': rows,
            'cols': cols,
            'crs': projection
        }

    # Extract metadata from data file
    metadata = read_metadata_from_gdal(data_file)
    
    # Validate dimensions
    if matrix.shape[0] != metadata['rows'] or matrix.shape[1] != metadata['cols']:
        raise ValueError("Matrix dimensions do not match data file dimensions")

    # Calculate the transform (from upper left corner)
    transform = from_origin(metadata['ulx'], metadata['uly'], metadata['pixel_size_x'], -metadata['pixel_size_y'])
    
    # Define metadata for the TIFF file
    tif_metadata = {
        'driver': 'GTiff',
        'count': 1,  # Number of bands
        'dtype': matrix.dtype.name,  # Data type of matrix
        'width': matrix.shape[1],
        'height': matrix.shape[0],
        'crs': metadata['crs'],
        'transform': transform
    }
    
    # Write the matrix to a TIFF file
    with rasterio.open(output_tif, 'w', **tif_metadata) as dst:
        dst.write(matrix, 1)  # Write the matrix to the first band

    # print(f'Data written to Tiff')
