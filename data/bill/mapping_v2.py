'''
Description
-----------
mapping_v2.py

    Burn-mapping using parallel computing.

    Requires NVidia GPU(s) for rendering.


What's new
----------
In this version, it's designed to take imagery of a single date with:

    Raw spectral bands.

    A mask as the hint. 
    
The algorithm will attempt to find the best mapping of burned areas.


Tradeoff
--------
In pre-post version (ver. 1), raster file with combination of pre and post as input, so changes are more detectable.


Strength
--------
This method is robust if we have access to just 1 most relevant date, not a date before the fire.


Syntax
------
python3 mapping_v2.py [Raster filename.bin] [Mask filename.bin] <- (could be a polygon mask)
'''

########### LIBRARIES ##################

from raster import Raster

from misc.general import (
    htrim_3d,
    extract_border,
    draw_border
)

from sampling import regular_sampling

import sys

import numpy as np

import ast

import time

########################################



class GUI_Settings:

    '''
    This setting contains parameters for model fitting.
    '''

    def __init__(self):

        self.rf_params = {
            'n_estimators': 100,    
            'max_depth': 15,
            'max_features': "sqrt", 
            'random_state': 42
        }

        #For clustering
        '''
        controled_ratio means we are not sure with our guess of the burn, so there might be more or less than
        what is actually sampled, currenly used for HDBSCAN min cluster size control. A positive value > 0.
        '''
        self.controlled_ratio = .8

        self.hdbscan_params = {
            'min_cluster_size': None,
            'min_samples': 20, # controls conservativeness
            'metric': 'euclidean'
        }



class GUI(GUI_Settings):

    def __init__(
            self,
            *,
            polygon_filename: str,
            image_filename: str,
            sample_size = 10_000,
            random_state = 123
    ):
        '''
        Initialized parameters
        ----------------------
        sample_size: number of points to be sampled from inside the polygon.
        '''

        super().__init__()
        
        #Default values
        self.polygon_filename = polygon_filename
        self.image_filename = image_filename

        #First plots
        self.embed_band_list = [1,2,4,6]
        self.img_band_list = [1,2,6]

        #Other settings
        self.sample_size = sample_size
        self.random_state = random_state

        #Init tasks
        self.load_image()
        self.load_polygon()
        self.sample_data()



    def load_image(
            self
    ):
        '''
        Load image data. Must be ENVI file.
        '''
        self.image = Raster(self.image_filename)

        self.image_dat = self.image.read_bands('all')



    def get_band_name(
            self,
            band_list: list
    ):
        '''
        Display real band name

        E.g: 1 -> B12, 2 -> B11.
        '''
        band_names = [self.image.band_name(i) for i in band_list]
        
        return band_names



    def load_polygon(
            self,
            border_thickness: int = 5
    ):
        '''
        Polygon needs to be rasterized using shapefile_rasterize_onto.py or equivalent.
        '''

        from exceptions.sen2 import PolygonException


        polygon = Raster(file_name=self.polygon_filename)

        if not polygon.is_polygon():
            #Check if this is a polygon.
            raise PolygonException(f"Not a polygon @ {self.polygon_filename}")
        
        #If it is a polygon, it has just 1 channel, so squeeze makes it a pretty 2D array.
        self.polygon = polygon

        #We use this to make a guess of the ratio.
        self.polygon_dat = polygon.read_bands('all').squeeze().astype(np.bool_)

        #extract border from the rasterized polygon.
        self.border = extract_border(
            mask = self.polygon_dat, 
            thickness = border_thickness
        )
    


    def sample_data(
            self
    ):
        '''
        For visualization of embedding space, sampling is essential.

        Use the original indices to determine which pixel is inside the polygon.
        '''

        self.sample_indices, self.samples = regular_sampling(
            raster_dat=self.image_dat,
            sample_size=self.sample_size,
            seed=self.random_state
        )

        #Which indices will be inside
        self.sample_in_polygon = ( self.polygon_dat.ravel() )[self.sample_indices].astype(np.bool_)

        '''
        The next part will be new, we will 
        '''
        self.guessed_burn_p = np.nanmean( self.polygon_dat )

        #We will later use this guess as a criteria for HDBSCAN
    


    def get_band_embed(
            self
    ):
        '''
        Uses TSNE algorithm to downsize n dim to just 2D

        Returns
        -------
        2D Embeddings
        '''

        from machine_learning.dim_reduce import tsne

        self.current_embed = tsne(
            self.samples, band_list=self.embed_band_list
        )

        self.current_band_name = ' | '.join(self.get_band_name(self.embed_band_list))

        embed_title = f"T-sne embedding | Samp. Sz: {self.sample_size} | Seed: {self.random_state}\n\n"
        embed_title += self.current_band_name

        return embed_title, self.current_embed
    


    def get_band_image(
            self,
            *,
            as_2D = False,
            filter_nan_with = None
    ):
        '''
        Image data but in all selected bands
        '''
        
        IMAGE = self.image_dat[..., [b-1 for b in self.embed_band_list]]

        if ( as_2D ):
            #Make it type row, nband
            n_bands = len( self.embed_band_list )
            IMAGE = IMAGE.reshape(-1, n_bands)

        if ( filter_nan_with is not None):
            #Pad NAN as another number 
            IMAGE = np.nan_to_num(IMAGE, nan=filter_nan_with)

        return IMAGE
    

    
    def get_shown_image(
            self,
            band_list = None
    ):
        '''
        Always use 'single' as key.

        Remember: band [1,2,3] looks different from [2,1,3].

        Band list item of 0 will be assumed to be 0.
        '''

        capped_band_list = self.img_band_list[:3] if band_list is None else band_list[:3]

        img_title = ' | '.join(self.get_band_name(self.img_band_list))

        if (len(self.img_band_list) > 3):

            img_title += f" | Warning! Showing first 3 bands (rgb) only."

        return img_title, htrim_3d( self.image_dat[..., [b - 1 for b in capped_band_list]] )
    


    '''
    CLASSIFICATION: THIS IS THE CORE of THE GUI.
    --------------
    '''

    def load_image_embed_RF(
            self
    ):
        '''
        Description
        -----------
        Use embedding of samples to transform all image.


        Returns
        -------
        Embeddings of the whole image derived from sampled embeddings.
        '''

        from machine_learning.trees import rf_regressor

        X = self.samples[..., [b-1 for b in self.embed_band_list]]
        y1 = self.current_embed[:, 0] #tsne1 from sample
        y2 = self.current_embed[:, 1] #tsne2 from sample

        embed_1 = rf_regressor(
            X, y1, **self.rf_params
        )

        embed_2 = rf_regressor(
            X, y2, **self.rf_params
        )

        input_img = self.get_band_image(
            as_2D = True, 
            filter_nan_with = 0.0
        )

        #Inference time
        img_embed_1 = embed_1.predict(input_img)
        img_embed_2 = embed_2.predict(input_img)

        #transformed image uses sampled embedding to transform the big picture
        transformed_img = np.column_stack((img_embed_1, img_embed_2))

        return transformed_img



    def map_burn(
            self
    ):
        '''
        Description
        -----------
        Uses current T-sne embedding to find the mapping.
        '''
        
        from machine_learning.cluster import (
            hdbscan_fit,
            hdbscan_approximate
        )

        #Step 1: prepare criteria for clustering.
        self.hdbscan_params['min_cluster_size'] = min(
            self.sample_size * self.guessed_burn_p * self.controlled_ratio,
            self.sample_size * ( 1-self.guessed_burn_p ) * self.controlled_ratio
        )

        t0 = time.time()

        transformed_img = self.load_image_embed_RF()

        cluster, _ = hdbscan_fit(
            self.current_embed, 
            **self.hdbscan_params
        )

        img_cluster, _ = hdbscan_approximate(
            transformed_img,
            cluster
        )

        print(f'Unique clusters: {np.unique(img_cluster)}')

        print(f'Mapping cost {time.time() - t0:.3f}s')

        return img_cluster
    


    def classify_cluster(
            self,
            cluster
    ):
        '''
        Clusters from HDBSCAN don't have names.

        Since we have mask as hint, we use it as information for the final labelling.

        Determines if cluster of 1 is burn or unburned. If most of the masked burn is closer to 1, then 
        
        we have some evidence that cluster 1 is burned.
        '''

        classification = np.full(self.polygon_dat.shape, False)

        masked_cluster = cluster[self.polygon_dat]

        if masked_cluster[masked_cluster != -1].mean() > 0.5:
            classification[cluster == 1] = True

        else:
            classification[cluster == 0] = True

        return classification



    def main(self):
        
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        embed_title, embed = self.get_band_embed()

        img_title, image = self.get_shown_image()

        ####### PLOT

        fig, (ax_tsne, ax_img) = plt.subplots(1, 2, figsize=(20, 12))

        ax_tsne.axis("off")
        ax_img.axis("off")

        tsne_colors = np.where(self.sample_in_polygon == 1, "red", "blue")

        tsne_plot = ax_tsne.scatter(
            embed[:, 0], 
            embed[:, 1],
            s = 5,
            c = tsne_colors,
            picker = 3 #enables clicking
        )

        ax_tsne.set_title( embed_title )

        ax_tsne.legend(
            handles = [
            mpatches.Patch(color="blue", label="Outside"),
            mpatches.Patch(color="red",  label="Inside"),
        ])


        img_plot = ax_img.imshow(
            draw_border(
                image, #Because band convention starts at 1, but index is from 0
                self.border
            )
        )
        
        ax_img.set_title(img_title)

        marker, = ax_img.plot([], [], "ro", markersize=6, fillstyle="none")

        # ----------------------------
        # 4. Click logic
        # ----------------------------
        hline = ax_img.axhline(0, color="red", linewidth=1.5, visible=False)
        vline = ax_img.axvline(0, color="red", linewidth=1.5, visible=False)

        W = self.image_dat.shape[1]

        def on_pick(event):

            k = event.ind[0]

            flat = self.sample_indices[k]

            r = flat // W
            c = flat %  W

            hline.set_ydata([r, r])
            vline.set_xdata([c, c])

            hline.set_visible(True)
            vline.set_visible(True)

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("pick_event", on_pick)

        from matplotlib.widgets import TextBox

        fig.subplots_adjust(
            left=0.03,
            right=0.97,
            bottom=0.05,
            top=0.88,
            wspace=0.05
        )


        '''
        Text box: T-sne
        '''
        ax_embed_box = fig.add_axes([0.15, 0.94, 0.1, 0.03])
        embed_textbox = TextBox(ax_embed_box, "TSNE Band list E.g 1,2,3 ")

        def on_submit_embed(txt):

            self.embed_band_list = ast.literal_eval(txt)

            #Set embed
            embed_title, embed = self.get_band_embed()

            # ---- update scatter IN PLACE ----
            tsne_plot.set_offsets( embed )
            ax_tsne.set_title( embed_title )

            fig.canvas.draw_idle()

        embed_textbox.on_submit(on_submit_embed)


        '''
        Text box: image
        '''
        ax_img_box = fig.add_axes([0.7, 0.94, 0.1, 0.03])
        img_textbox = TextBox(ax_img_box, "Image Band list E.g 1,2,3 ")

        def on_submit_img(txt):

            #Shows first 3 only
            self.img_band_list = ast.literal_eval(txt)

            img_title, image = self.get_shown_image()

            img_plot.set_data(
                draw_border(
                    image,
                    self.border
                )
            )

            ax_img.set_title( img_title )

            # ---- update image IN PLACE ----

            fig.canvas.draw_idle()

        img_textbox.on_submit(on_submit_img)


        '''
        Text box: Mapping
        '''
        from matplotlib.widgets import Button

        ax_mapping = fig.add_axes([0.3, 0.94, 0.1, 0.03])
        mapping_btn = Button(ax_mapping, "Classify")

        def on_submit_mapping(event):

            print(f'RUNNING ... Mapping on band list: {self.embed_band_list}')

            #Get clusters
            img_cluster = self.map_burn()

            img_cluster = img_cluster.reshape(self.image._ySize, self.image._xSize)

            #Get true classification
            classification = self.classify_cluster(img_cluster)

            fig2 = plt.figure(figsize=(12, 12))

            ax2 = fig2.add_subplot(111)
    
            ax2.imshow(
                classification, cmap = 'gray'
            )

            ax2.set_title(f'Burn Mapping on {len(self.embed_band_list)} bands\n{self.current_band_name}')

            ax2.axis("off")

            print(f'DONE! Mapping on band list: {self.embed_band_list}')

            plt.tight_layout()

            fig2.show()

        mapping_btn.on_clicked(on_submit_mapping)


        plt.show()



if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Needs 1 raster file and 1 polygon file")
        sys.exit(1)

    image_filename = sys.argv[1]
    polygon_filename = sys.argv[2]

    agent = GUI(
        polygon_filename=polygon_filename,
        image_filename=image_filename
    )

    agent.main()