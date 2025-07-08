'''20250707 barc_plot.py based on code by Sterling von Dehn
'''
from misc import err, read_binary, args, read_hdr, hdr_fn, band_names
import matplotlib.pyplot as plt
import numpy as np
import os

import rasterio
from rasterio.transform import from_origin
from osgeo import gdal

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


def barc_class_plot(dNBR, start_date, end_date, title='Not given'):
    '''
    Plots the BARC 256 burn severity of the provided dNBR and saves it as a png
    '''

    scaled_dNBR = (dNBR*1000+275)/5 #scalling dNBR
    class_plot = np.zeros((len(scaled_dNBR),len(scaled_dNBR[0])))
    un_tot = 0
    low_tot = 0
    med_tot = 0
    high_tot = 0
    for i in range(len(scaled_dNBR)): #making classifications
        for j in range(len(scaled_dNBR[0])):
            if scaled_dNBR[i][j] < 76:
                class_plot[i][j] = 1
                un_tot += 1
            elif 76 <= scaled_dNBR[i][j] < 110:
                class_plot[i][j] = 2
                low_tot += 1
            elif 110 <= scaled_dNBR[i][j] < 187:
                class_plot[i][j] = 3
                med_tot += 1
            elif np.isnan(scaled_dNBR[i][j]):
                class_plot[i][j] = float('nan')
            else:
                class_plot[i][j] = 4
                high_tot += 1

    #calculating percentages
    tot = un_tot+low_tot+med_tot+high_tot
    un_per = round(100*un_tot/tot,1)
    low_per = round(100*low_tot/tot,1)
    med_per = round(100*med_tot/tot,1)
    high_per = round(100*high_tot/tot,1)

    if not os.path.exists(title):
        os.mkdir(title)
    #plotting
    cmap = matplotlib.colors.ListedColormap(['green','yellow','orange','red'])
    plt.figure(figsize=(15,15))
    plt.imshow(class_plot,vmin=1,vmax=4,cmap=cmap)
    plt.title(f'BARC 256 burn severity, start date:{start_date}, end date:{end_date}')
    plt.scatter(np.nan,np.nan,marker='s',s=100,label=f'Unburned {un_per}%',color='green')
    plt.scatter(np.nan,np.nan,marker='s',s=100,label=f'Low {low_per}%' ,color='yellow')
    plt.scatter(np.nan,np.nan,marker='s',s=100,label=f'Medium {med_per}%',color='orange')
    plt.scatter(np.nan,np.nan,marker='s',s=100,label=f'High {high_per}%',color='red')
    plt.legend(fontsize="20")
    plt.tight_layout()
    plt.savefig(f'{title}/{end_date}_BARC_classification.png')
    plt.close()
    print('+w', f'{title}/{end_date}_BARC_classification.png')
    return class_plot



if __name__ == '__main__':
    if len(args) < 3:
        err("barc plot: [pre-image] [post-image] # files in ENVI format")

    ncol, nrow, nband = read_hdr(hdr_fn(args[1]))
    ncol2, nrow2, nband2 = read_hdr(hdr_fn(args[2]))
 
    if ncol != ncol2 or nrow != nrow2:
        err("pre and post image dimensions must match")

    # band index determination 
    def band_ix(band_names, pattern):
        ix = None
        for i in range(len(band_names)):
            if len(band_names[i].split(pattern)) > 1:
                ix = i
        return ix

    band_names_1 = band_names(hdr_fn(args[1]))
    band_names_2 = band_names(hdr_fn(args[2]))
    print(band_names_1, band_names_2)

    B08ix1 = band_ix(band_names_1, 'B8')
    B08ix2 = band_ix(band_names_2, 'B8')
    B12ix1 = band_ix(band_names_1, 'B12')
    B12ix2 = band_ix(band_names_2, 'B12')

    pre_img = read_binary(args[1])
    pst_img = read_binary(args[2])

    nrow, ncol, nband = int(nrow), int(ncol), int(nband)
    npx = nrow * ncol
    B08_1 = pre_img[3][B08ix1 * npx: (B08ix1 + 1) * npx]
    B12_1 = pre_img[3][B12ix1 * npx: (B12ix1 + 1) * npx]
    B08_2 = pst_img[3][B08ix2 * npx: (B08ix2 + 1) * npx]
    B12_2 = pst_img[3][B12ix2 * npx: (B12ix2 + 1) * npx]

    def NBR(B08, B12):
        return (B08-B12)/(B08+B12)#calculating NBR

    NBR_pre, NBR_post = NBR(B08_1, B12_1), NBR(B08_2, B12_2)
       
    dNBR = NBR_pre - NBR_post #calculating dNBR

    plt.figure(figsize=(15,15))
    plt.imshow(NBR_pre.reshape(nrow, ncol))
    plt.title('NBR: pre-image: ' + str(args[1]))
    plt.tight_layout()
    plt.savefig('nbr_pre.png')

    plt.figure(figsize=(15,15))
    plt.imshow(NBR_post.reshape(nrow, ncol))
    plt.title('NBR: post-image: ' + str(args[2]))
    plt.tight_layout()
    plt.savefig('nbr_post.png')

    plt.figure(figsize=(15,15))
    plt.imshow(dNBR.reshape(nrow, ncol))
    plt.title('dNBR: pre-image ' + str(args[1]) + ' post-image: ' + str(args[2]))
    plt.tight_layout()
    plt.savefig('dnbr.png')

    dNBR = dNBR.reshape(nrow, ncol)
    start_date = '2025xxxx'
    end_date = '2025xxxx'
    class_map = barc_class_plot(dNBR, start_date, end_date, title='Not given')

    end_file = str(argv[2])
    write_matrix_to_tif(class_map, end_file, 'barc.tif')
