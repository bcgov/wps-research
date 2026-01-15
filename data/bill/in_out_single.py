'''
interactive_in_out.py
'''

import matplotlib.pyplot as plt

from raster import Raster

from sampling import in_out_sampling

import numpy as np

import sys

import ast


seed = 42


if __name__ == '__main__':

    #handling argv
    if len(sys.argv) < 2:
        print("Needs 1 raster file")
        sys.exit(1)

    raster_filename = sys.argv[1]


    band_lst = [1, 2, 3] #Default
    if len(sys.argv) > 2:
        band_lst = ast.literal_eval(sys.argv[2])


    polygon_filename = 'rasterized_0000.bin'

    #Read Raster data for pixel referencing
    raster = Raster(file_name=raster_filename)
    raster_dat = raster.readBands_and_trim(band_lst=band_lst)


    in_sample_size = 100 #Sample size inside the perimeter

    original_indices, samples = in_out_sampling(
        raster_filename=raster_filename,
        polygon_filename=polygon_filename,
        in_sample_size = in_sample_size
    )


    #TSNE dimensionality reduction
    from dim_reduce import tsne


    #We will use this param setting
    tsne_params = {
        'n_components': 2,
        'perplexity': 30,
        'learning_rate': "auto",
        'init': "pca",
        'random_state': seed
    }

    Y = tsne(
        X = samples,
        params=tsne_params
    )

    ############ Interactive Map ############################

    fig, (ax_tsne, ax_img) = plt.subplots(1, 2, figsize=(20, 8))

    sc_in = ax_tsne.scatter(
        Y[:in_sample_size, 0], 
        Y[:in_sample_size, 1],
        s=30,
        c='red',
        label='Inside',
        picker=3  # ← this enables clicking
    )

    sc_out = ax_tsne.scatter(
        Y[in_sample_size:, 0], 
        Y[in_sample_size:, 1],
        s=30,
        c='blue',
        label='Outside',
        picker=3
    )

    ax_tsne.set_title("t-SNE space")

    ax_tsne.legend()

    #Right side of the plot, the main image (let parameter determine different band combination)

    img_plot = ax_img.imshow(raster_dat)
    
    ax_img.set_title(f"Map view, band = {band_lst}")

    ################################################################

    marker, = ax_img.plot([], [], "ro", markersize=6, fillstyle="none")

    # ----------------------------
    # 4. Click logic
    # ----------------------------
    hline = ax_img.axhline(0, color="r", linewidth=1, visible=False)
    vline = ax_img.axvline(0, color="r", linewidth=1, visible=False)

    def on_pick(event):

        k = event.ind[0]

        flat = original_indices[k]

        W = raster._xSize

        r = flat // W
        c = flat %  W

        hline.set_ydata([r, r])
        vline.set_xdata([c, c])

        hline.set_visible(True)
        vline.set_visible(True)

        fig.canvas.draw_idle()

        
        label = "INside" if (k < in_sample_size) else "OUTside"

        print('-' * 30)
        print(f'Picked {label}')




    fig.canvas.mpl_connect("pick_event", on_pick)
    plt.show()