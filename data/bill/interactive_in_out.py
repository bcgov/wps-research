'''

'''


if __name__ == '__main__':

    from sklearn.manifold import TSNE

    from sklearn.preprocessing import StandardScaler

    import matplotlib.pyplot as plt

    from polygon import split_in_out

    from sampling import row_sampling

    from misc.general import htrim_3d

    import numpy as np



    raster_file_name = 'S2A_MSIL2A_20251009T193831_N0511_R142_T09UYU_20251009T234613_cloudfree.bin_MRAP_C11659.bin_crop.bin'

    polygon_file_name = 'rasterized_0000.bin'


    raster_data, _, post_inside, post_outside = split_in_out(
        raster_filename=raster_file_name,
        polygon_filename=polygon_file_name
    )

    out_in_ratio = len(post_outside) / len(post_inside)


    #Sample size determination

    in_sample_size = 10
    out_sample_size = int(in_sample_size * out_in_ratio)

    #Sample now
    orig_in_idx,  inside_samples  = row_sampling(post_inside,  size=in_sample_size, filter_nan=True)
    orig_out_idx, outside_samples = row_sampling(post_outside, size=out_sample_size, filter_nan=True)

    samples = np.vstack([
        inside_samples,
        outside_samples
    ])

    original_indices = np.concatenate([orig_in_idx, orig_out_idx])


    #TSNE

    X_s = StandardScaler().fit_transform(samples)


    #We will use this param setting for all data
    tsne_params = {
        'n_components': 2,
        'perplexity': 30,
        'learning_rate': "auto",
        'init': "pca",
        'random_state': 42
    }

    tsne = TSNE(**tsne_params)

    Y = tsne.fit_transform(X_s)


    ############ Interactive Map ############################

    fig, (ax_tsne, ax_img) = plt.subplots(1, 2, figsize=(12, 5))

    sc_in = ax_tsne.scatter(
        Y[: in_sample_size, 0], 
        Y[: in_sample_size, 1],
        s=10,
        c='red',
        label='Inside',
        picker=3  # ← this enables clicking
    )

    sc_out = ax_tsne.scatter(
        Y[in_sample_size:, 0], 
        Y[in_sample_size:, 1],
        s=10,
        c='blue',
        label='Outside',
        picker=3  # ← this enables clicking
    )

    ax_tsne.set_title("t-SNE space")

    ax_tsne.legend()

    

    img_plot = ax_img.imshow(htrim_3d(raster_data[..., :3]))
    
    ax_img.set_title("Map view")

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

        W = 1354


        r = flat // W
        c = flat % W

        hline.set_ydata([r, r])
        vline.set_xdata([c, c])

        hline.set_visible(True)
        vline.set_visible(True)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)
    plt.show()