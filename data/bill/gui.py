'''
interactive_in_out.py (parallel dimensionality reduction)


How this works?
---------------
Read all
'''

########### LIBRARIES ##################
import matplotlib.pyplot as plt

from raster import Raster

from misc.general import (
    htrim_3d,
    extract_border,
    draw_border
)

from sampling import in_out_sampling

import numpy as np

import os

import sys

import ast

from dim_reduce import (
    tsne,
    pca,
    parDimRed
)


class GUI:

    def __init__(
            self,
            random_state = 123,
            in_sample_size = 100,
            method = 'tsne'
    ):
        '''
        Initialized parameters
        ----------------------
        in_sample_size: number of points to be sampled from inside the polygon.

        *out_sample_size: will automatically calculate using true ratio of in-out.
        '''
        
        #Default values
        self.random_state = random_state
        self.in_sample_size = in_sample_size
        self.method = method



    def __load_raster(
            self,
            filename: str
    ):
        '''
        Load raster data. Must be ENVI file or tiff file.
        '''
        self.raster = Raster(file_name = filename)

        return



    def __load_polygon(
            self,
            filename: str,
            border_thickness: int = 8
    ):
        '''
        Polygon needs to be rasterized using shapefile_rasterize_onto.py or equivalent.
        '''

        from exceptions.sen2 import PolygonException


        polygon = Raster(file_name=polygon_filename)

        if not polygon.is_polygon(polygon):
            #Check if this is a polygon.
            raise PolygonException(f"Not a polygon @ {filename}")
        

        #If it is a polygon, it has just 1 channel, so squeeze makes it a pretty 2D array.
        self.polygon = polygon

        #extract border
        self.border = extract_border(
            mask=polygon_dat.squeeze(), 
            thickness=border_thickness
        )

        return
    


    def __sampling_in_out(
            self
    ):
        '''
        For visualization of embedding space, sampling is essential.
        '''

        original_indices, samples, out_in_ratio = in_out_sampling(
            raster_dat=self.raster.read_bands('all'),
            polygon_dat=self.polygon.read_bands('all'),
            in_sample_size = self.in_sample_size
        )

        self.out_sample_size = int( self.in_sample_size * out_in_ratio )

        return original_indices, samples




if __name__ == '__main__':

    ### SOME DEFAULT VALUE, CAN SET AS INPUT LATER #######

    ############### handling argv #######################


    if len(sys.argv) < 3:
        print("Needs 1 raster file and 1 polygon file")
        sys.exit(1)

    raster_filename = sys.argv[1]

    polygon_filename = sys.argv[2]

    if len(sys.argv) > 3:

        band_lst = ast.literal_eval(sys.argv[3])

    method_name = 'tsne'  #Argument as well


    #Read Raster data for pixel referencing
    raster = Raster(file_name=raster_filename)
    raster_dat = raster.read_bands(band_lst='all')

    #Read Polygon data for pixel referencing
    polygon = Raster(file_name=polygon_filename)
    polygon_dat = polygon.read_bands(band_lst=[1])

    ##############Sampling, the sample contains all bands in the data #############

    original_indices, samples, out_in_ratio = in_out_sampling(
        raster_filename=raster_filename,
        polygon_filename=polygon_filename,
        in_sample_size = in_sample_size
    )

    out_sample_size = int( in_sample_size * out_in_ratio )

    ############ Read the border of polygon ###############

    border = extract_border(polygon_dat.squeeze(), thickness=8)

    ############ Load data from cache ######################

    #Check if we already have the data stored.
    CACHE_PATH = f'caching/mt={method_name}_sz={in_sample_size}_rs={seed}_timestamp={raster.acquisition_timestamp}.npz'

    if os.path.exists(CACHE_PATH):

        print("Loading cached embeddings...")

        cache = dict(np.load(CACHE_PATH, allow_pickle=True))

    else:

        '''
        Write new data.
        '''

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

        method_dict = {'tsne': tsne, 'pca': pca}

        #Prepare tasks
        tasks = [
            (
                b, samples[..., [bb - 1 for bb in b]], method_dict[method_name], tsne_params
            )
            
            for b in band_combinations
        ]

        cache = parDimRed(tasks)

        np.savez(CACHE_PATH, **cache)

        print(f"Data saved to {CACHE_PATH}.")

    X = cache[str(band_lst)] #Choose data from cache dict based on band list

    ############ Interactive Map ############################

    fig, (ax_tsne, ax_img) = plt.subplots(1, 2, figsize=(20, 8))

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

    ax_tsne.set_title(f"{method_name} space | Sample Size of {in_sample_size} / {out_sample_size} | Random State: {seed}")

    ax_tsne.legend()

    #Right side of the plot, the main image (let parameter determine different band combination)

    img_plot = ax_img.imshow(

        draw_border(
            htrim_3d( raster_dat[..., [b - 1 for b in band_lst]] ), #Because band convention starts at 1, but index is from 0
            border
        )

    )
    
    ax_img.set_title(f"Band: {band_lst}")

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


    ############# ADJUST DATA SHOWN #####################
    from matplotlib.widgets import TextBox

    fig.subplots_adjust(top = 2)

    # add textbox
    ax_box = fig.add_axes([0.35, 0.9, 0.3, 0.04])
    textbox = TextBox(ax_box, "Band list..e.g [1,2,3]: ")

    def on_submit(txt):

        band_lst = ast.literal_eval(txt)

        try:
            #Set TNSE
            new_X = cache[str(band_lst)]

        except Exception:

            raise KeyError("This band combination is not in cache.")

        img_plot.set_data(
            draw_border(
                htrim_3d(raster_dat[..., [b - 1 for b in band_lst]]),
                border
            )
        )

        ax_img.set_title(f"Band: {txt}")

        # ---- update scatter IN PLACE ----
        sc_in.set_offsets(new_X[:in_sample_size])
        sc_out.set_offsets(new_X[in_sample_size:])

        fig.canvas.draw_idle()

    textbox.on_submit(on_submit)


    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()
