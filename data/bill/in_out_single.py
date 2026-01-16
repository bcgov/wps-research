'''
interactive_in_out.py (parallel dimensionality reduction)


How this works?
---------------
Read all
'''


import matplotlib.pyplot as plt

from raster import Raster

from misc.general import htrim_3d

from sampling import in_out_sampling

import numpy as np

import sys

import ast


seed = 42

#Sample size inside the perimeter
in_sample_size = 10

if __name__ == '__main__':

    #handling argv
    if len(sys.argv) < 3:
        print("Needs 1 raster file and 1 polygon file")
        sys.exit(1)

    raster_filename = sys.argv[1]

    polygon_filename = sys.argv[2]

    if len(sys.argv) > 3:

        band_lst = ast.literal_eval(sys.argv[3])

    method = 'tsne' #Argument as well




    #Read Raster data for pixel referencing
    raster = Raster(file_name=raster_filename)
    raster_dat = raster.read_bands(band_lst='all')

    #Sampling, the sample contains all bands in the data
    original_indices, samples, out_in_ratio = in_out_sampling(
        raster_filename=raster_filename,
        polygon_filename=polygon_filename,
        in_sample_size = in_sample_size
    )

    out_sample_size = int( in_sample_size * out_in_ratio )

    #### TSNE dimensionality reduction #####
    import os

    #Check if we already have the data stored.
    CACHE_PATH = f'caching/{method}_{in_sample_size}.npz'

    if os.path.exists(CACHE_PATH):

        print("Loading cached embeddings...")

        cache = dict(np.load(CACHE_PATH, allow_pickle=True))

    else:

        from dim_reduce import *

        #We will use this param setting
        tsne_params = {
            'n_components': 2,
            'perplexity': 30,
            'learning_rate': "auto",
            'init': "pca",
            'random_state': seed
        }

        #Find all combinations of bands here. I am quite lazy so let's hardcode it.
        #At least 3 bands (who projects 2 bands onto 2 bands? right?)
        band_combinations = [
            [1,2,3],
            [1,2,4],
            [1,3,4],
            [2,3,4],
            [1,2,3,4]
        ]

        #Prepare tasks
        tasks = [
            (
                b, samples[..., [b - 1 for b in band_lst]], tsne, tsne_params
            )
            
            for b in band_combinations
        ]

        cache = parDimRed(tasks)

        np.savez(CACHE_PATH, **cache)

        print(f"Data saved to {CACHE_PATH}.")




    ############ Interactive Map ############################

    fig, (ax_tsne, ax_img) = plt.subplots(1, 2, figsize=(20, 8))

    #Choose data from tsne_cache based on band list

    X = cache[str(band_lst)]

    sc_in = ax_tsne.scatter(
        X[:in_sample_size, 0], 
        X[:in_sample_size, 1],
        s = 30,
        c='red',
        label='Inside',
        picker=3  # ← this enables clicking
    )

    sc_out = ax_tsne.scatter(
        X[in_sample_size:, 0], 
        X[in_sample_size:, 1],
        s=30,
        c='blue',
        label='Outside',
        picker=3
    )

    ax_tsne.set_title(f"t-SNE space | Sample Size of {in_sample_size} / {out_sample_size} | Random State: {seed}")

    ax_tsne.legend()

    #Right side of the plot, the main image (let parameter determine different band combination)

    img_plot = ax_img.imshow(
        htrim_3d( raster_dat[..., [b - 1 for b in band_lst]] ) #Because band convention starts at 1, but index is from 0
    )
    
    ax_img.set_title(f"{band_lst}")

    ################################################################

    marker, = ax_img.plot([], [], "ro", markersize=6, fillstyle="none")

    # ----------------------------
    # 4. Click logic
    # ----------------------------
    hline = ax_img.axhline(0, color="red", linewidth=2, visible=False)
    vline = ax_img.axvline(0, color="red", linewidth=2, visible=False)

    W = raster_dat.shape[1]

    def on_pick(event):

        k = event.ind[0]

        if (event.artist is sc_in):
            crosshair_colour = "red"
            flat = original_indices[:in_sample_size][k]

        elif (event.artist is sc_out):
            crosshair_colour = "blue"
            flat = original_indices[in_sample_size:][k]

        r = flat // W
        c = flat %  W

        hline.set_ydata([r, r])
        vline.set_xdata([c, c])

        hline.set_visible(True)
        vline.set_visible(True)

        fig.canvas.draw_idle()

        hline.set_color(crosshair_colour)
        vline.set_color(crosshair_colour)



    fig.canvas.mpl_connect("pick_event", on_pick)

    plt.tight_layout()
    plt.show()
