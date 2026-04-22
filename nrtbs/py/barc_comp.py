'''
plots a comparison between FAIB and NRTBS BARC plots. Takes fire number, FAIB_BARC_start_date, NRTBS_BARC_start_date, composite_start_date, end_date
$ python3 barc_comp.py N51117 20240712 20240715 20240629 20240801
'''
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from rasterio.plot import show
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from numpy.ma import masked_where
import geopandas as gpd
from misc import run, args

def comp_tiff(fire_num, start_date1, start_date2, start_date3, end_date, historical_perimeters=None):
    '''
    Compares BARC plots
    '''
    #getting data files
    faib_tiff = f'../data/{fire_num}/barc/BARC_{fire_num}_{start_date1}_{end_date}_S2_clip.tif'
    nrtbs_tiff_pre = f'{fire_num}_barcs/{end_date}_barc.tif'
    #trimming NRTBS tiff
    trim_tif_to_shapefile(nrtbs_tiff_pre, fire_num, f'{fire_num}_barcs/cut.tif', historical_perimeters)
    run(f'python3 raster_project_onto.py {fire_num}_barcs/cut.tif {faib_tiff} {fire_num}_barcs/{fire_num}_projected.tif true')
    nrtbs_tiff = f'{fire_num}_barcs/{fire_num}_projected.tif'

    #reading files
    with rasterio.open(nrtbs_tiff) as src:
        nrtbs_barc = src.read(1)
    with rasterio.open(faib_tiff) as src:
        faib_barc = src.read(1)

    faib_barc[faib_barc == 9] = 0 #setting nans to 0

    #Subtracting results
    sub_barc = abs(nrtbs_barc - faib_barc)

    #plotting
    cmap = matplotlib.colors.ListedColormap(['green','yellow','orange','red'])
    cmap_sub = matplotlib.colors.ListedColormap(['blue', 'green','yellow','orange']) 
    fig, axs = plt.subplots(1, 3, figsize=(20, 15))
    axs[0].imshow(nrtbs_barc,vmin=1,vmax=4,cmap=cmap)
    axs[0].set_title(f'MRAP NRTBS   start:{start_date2}, end:{end_date}')
    axs[0].scatter(np.nan,np.nan,marker='s',s=100,label=f'Unburned' ,color='green')
    axs[0].scatter(np.nan,np.nan,marker='s',s=100,label=f'Low' ,color='yellow')
    axs[0].scatter(np.nan,np.nan,marker='s',s=100,label=f'Medium',color='orange')
    axs[0].scatter(np.nan,np.nan,marker='s',s=100,label=f'High',color='red')
    axs[0].set_xlabel(f'Composite date range: {start_date3} - {end_date}')
    axs[0].legend()
    axs[1].imshow(faib_barc,vmin=1,vmax=4,cmap=cmap)
    axs[1].set_title(f'FAIB GEE   start:{start_date1}, end:{end_date}')
    axs[1].scatter(np.nan,np.nan,marker='s',s=100,label=f'Unburned' ,color='green')
    axs[1].scatter(np.nan,np.nan,marker='s',s=100,label=f'Low' ,color='yellow')
    axs[1].scatter(np.nan,np.nan,marker='s',s=100,label=f'Medium',color='orange')
    axs[1].scatter(np.nan,np.nan,marker='s',s=100,label=f'High',color='red')
    axs[1].legend()
    axs[2].imshow(sub_barc, vmin=0,vmax=4,cmap=cmap_sub)
    axs[2].set_title('|MRAP NRTBS - FAIB GEE|')
    axs[2].scatter(np.nan,np.nan,marker='s',s=100,label=f'0',color='blue')
    axs[2].scatter(np.nan,np.nan,marker='s',s=100,label=f'1',color='green')
    axs[2].scatter(np.nan,np.nan,marker='s',s=100,label=f'2' ,color='yellow')
    axs[2].scatter(np.nan,np.nan,marker='s',s=100,label=f'3',color='orange')
    axs[2].legend()
    fig.suptitle(f'{fire_num} BARC comparison', fontsize=25)
    plt.tight_layout()
    plt.savefig(f'{fire_num}_barcs/{fire_num}_comp.png')

def trim_tif_to_shapefile(tif_path, fire_num, output_path, historical_perimeters=None):
    '''
    cuts a tiff file to a fire perimeter
    '''
    # Load the shapefile
    shapefile_path = 'prot_current_fire_polys.shp' if historical_perimeters is None else historical_perimeters
    perims = gpd.read_file(shapefile_path)
    fire_number_string = 'FIRE_NUMBE' if 'FIRE_NUMBE' in perims else 'FIRE_NUM'
    shapes = perims[perims[fire_number_string] == fire_num]
    
    # Ensure the shapefile and TIFF are in the same CRS
    with rasterio.open(tif_path) as src:
        tif_crs = src.crs
        # Reproject shapefile to match the CRS of the TIFF
        shapes = shapes.to_crs(tif_crs)
        
        # Convert shapes to geometries in the form expected by rasterio
        geometries = [geom['geometry'] for _, geom in shapes.iterrows()]
        
        # Open the TIFF file
        with rasterio.open(tif_path) as src:
            # Mask the TIFF file using the shapefile geometries
            out_image, out_transform = mask(src, geometries, crop=True)
            
            # Update the metadata with new dimensions and transform
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "count": out_image.shape[0],
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            # Write the trimmed TIFF file to disk
            with rasterio.open(output_path, 'w', **out_meta) as dest:
                dest.write(out_image)

if __name__ == "__main__":
    comp_tiff(args[1],args[2],args[3], args[4], args[5])
