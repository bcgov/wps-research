'''
interactive_in_out.py
'''


from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from raster import Raster

from polygon import split_in_out

from sampling import row_sampling

from change_detection import change_detection

from misc.general import htrim_3d

import numpy as np

import sys

import ast




seed = 42



if __name__ == '__main__':

    band_lst = [1, 2, 3] #Default
    #handling argv
    if len(sys.argv) > 1:
        band_lst = ast.literal_eval(sys.argv[1])



    raster_file_name_pre = 'S2C_MSIL2A_20250609T192931_N0511_R142_T09UYU_20250610T002612_cloudfree.bin_MRAP_C11659.bin_crop.bin'

    raster_file_name_post = 'S2A_MSIL2A_20251009T193831_N0511_R142_T09UYU_20251009T234613_cloudfree.bin_MRAP_C11659.bin_crop.bin'

    polygon_file_name = 'rasterized_0000.bin'

    #Read Raster data for pixel referencing
    raster_pre  = Raster(file_name=raster_file_name_pre)
    raster_post = Raster(file_name=raster_file_name_post)

    raster_pre_dat = raster_pre.read_bands(band_lst=band_lst)
    raster_post_dat = raster_post.read_bands(band_lst=band_lst)

    #Change Detection
    change = change_detection(
        pre_X=raster_pre_dat,
        post_X=raster_post_dat
    )

    #Extract inside and outside matrix (split)
    inside, in_indices, outside, out_indices = split_in_out(
        raster_dat = change,
        polygon_filename = polygon_file_name
    )

    
    #Ratio to maintain the true population proportion
    out_in_ratio = len(outside) / len(inside)


    #Sample size determination

    in_sample_size = 100
    out_sample_size = int(in_sample_size * out_in_ratio)

    #Sample now
    in_idx_samples,  inside_samples  = row_sampling(X=inside,  
                                                 size=in_sample_size, 
                                                 original_indices=in_indices,
                                                 filter_nan=True,
                                                 seed = seed)
    
    out_idx_samples, outside_samples = row_sampling(X=outside, 
                                                 size=out_sample_size, 
                                                 original_indices=out_indices,
                                                 filter_nan=True,
                                                 seed = seed)


    samples = np.vstack([
        inside_samples,
        outside_samples
    ])

    original_indices = np.concatenate([in_idx_samples, 
                                       out_idx_samples])


    #TSNE dimensionality reduction
    X_s = StandardScaler().fit_transform(samples)


    #We will use this param setting
    tsne_params = {
        'n_components': 2,
        'perplexity': 30,
        'learning_rate': "auto",
        'init': "pca",
        'random_state': seed
    }

    tsne = TSNE(**tsne_params)

    Y = tsne.fit_transform(X_s)


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

    ax_tsne.set_title("t-SNE space for change detection")

    ax_tsne.legend()

    #Right side of the plot, the main image (let parameter determine different band combination)

    img_plot = ax_img.imshow(
        htrim_3d(change[..., :3])
    )
    
    ax_img.set_title(f"Map view for change detection, band = {band_lst}")

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

        W = raster_pre._xSize #either pre or post is fine

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