'''
Misc file to work with photos and videos
'''

from raster import Raster
from .files import iter_files
import numpy as np
import matplotlib.pyplot as plt

def save_png_same_dir(
        dir: str,
        band_list: list[int] = None,
        p_trim: int = 1
):
    
    if band_list and len(band_list) > 3:
        raise ValueError(f'RGB accepts 3 bands, received {len(band_list)}')
    
    if band_list is None:

        band_list = [1,2,3]

    lo = hi = None

    for filename in iter_files(dir, '.bin'):

        raster = Raster(filename)

        raster_dat = raster.read_bands(band_list)

        if lo is None: lo, hi = np.nanpercentile(raster_dat, p_trim, axis=(0, 1)), np.nanpercentile(raster_dat, 100 - p_trim, axis=(0, 1))

        raster_dat = np.clip((raster_dat - lo) / (hi - lo), 0, 1)

        # Generate PNG filename
        png_path = filename.replace(".bin", ".png")

        # Save PNG
        if raster_dat.ndim == 1:
            plt.imsave(png_path, raster_dat, cmap="gray")

        else:
            plt.imsave(png_path, raster_dat)
            
        print(f"Saved {png_path}")


if __name__ == "__main__":

    save_png_same_dir(
        'mrap'
    )




    

    

        

        
    