'''20260217 burn_mapping.py adapted from ../data/bill/burn_mapping.py
UPDATE
------
Polygon is not required as the hint anymore.

There will be list of options for criterion you can choose as the hint.

Once you are happy with the hint, click 'Apply Hint' and let the model works the mapping out.


Description
-----------
Burn-mapping using parallel computing.

Requires NVidia GPU(s) for rendering.


Inputs
------
In this version, it's designed to take imagery of a single date with:

    Raw spectral bands.

    A mask as the hint. [optional]
    
The algorithm will attempt to find the best mapping of burned areas.


Syntax
------
>> python3 mapping.py [Raster filename.bin] [Mask filename.bin] <- (could be a polygon mask)
'''

########### My LIBRARIES ##################

from raster import Raster

from misc.sen2 import writeENVI

from misc.general import (
    htrim_3d,
    extract_border,
    draw_border
)

from sampling import regular_sampling

########### Built-in LIBRARIES ###########

import sys

import os

import ast

import time

import numpy as np

########### Matplotlib ###################

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

########################################


class BandListPicker:
    """
    Multi-select list widget for band selection.
    Click items to toggle selection. Selected items are highlighted.
    """
    def __init__(self, ax, band_names, initially_selected_indices, title="Select Bands"):
        self.ax = ax
        self.band_names = band_names
        self.selected_indices = set(initially_selected_indices)
        self.title = title
        
        # Visual properties
        self.item_height = 0.8 / len(band_names) if len(band_names) > 0 else 0.1
        self.bg_color_selected = 'lightblue'
        self.bg_color_unselected = 'white'
        self.text_color = 'black'
        
        # Setup axes
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        
        # Title
        self.title_obj = self.ax.text(0.5, 0.95, title, 
                                       va='top', ha='center', 
                                       fontsize=10, weight='bold')
        
        # Create list items
        self.items = []  # List of (rectangle, text, index)
        for i, band_name in enumerate(band_names):
            y_bottom = 0.9 - (i + 1) * self.item_height
            
            # Background rectangle
            is_selected = i in self.selected_indices
            rect = Rectangle((0.05, y_bottom), 0.9, self.item_height * 0.95,
                            facecolor=self.bg_color_selected if is_selected else self.bg_color_unselected,
                            edgecolor='gray', linewidth=1)
            self.ax.add_patch(rect)
            
            # Text label
            text = self.ax.text(0.1, y_bottom + self.item_height * 0.5, 
                               band_name,
                               va='center', ha='left', fontsize=9,
                               color=self.text_color)
            
            self.items.append((rect, text, i))
        
        # Connect click event
        self.cid = self.ax.figure.canvas.mpl_connect('button_press_event', self._on_click)
    
    def _on_click(self, event):
        """Toggle selection on click"""
        if event.inaxes != self.ax:
            return
        
        # Find which item was clicked
        for rect, text, idx in self.items:
            # Check if click is within rectangle bounds
            if rect.contains(event)[0]:
                # Toggle selection
                if idx in self.selected_indices:
                    self.selected_indices.remove(idx)
                    rect.set_facecolor(self.bg_color_unselected)
                else:
                    self.selected_indices.add(idx)
                    rect.set_facecolor(self.bg_color_selected)
                
                self.ax.figure.canvas.draw_idle()
                break
    
    def get_selected_indices(self):
        """Return list of selected band indices (0-based)"""
        return sorted(list(self.selected_indices))
    
    def get_selected_band_indices(self):
        """Return list of selected band indices (1-based for the actual band numbers)"""
        return [i + 1 for i in self.get_selected_indices()]


class GUI_Settings:

    '''
    This setting contains parameters for model fitting.
    '''

    def __init__(self):

        #for file saving
        self.save_dir = './mapped_burn'

        #For TSNE embedding.
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
        self.controlled_ratio = .5

        self.hdbscan_params = {
            'min_cluster_size': None,
            'min_samples': 20, # controls conservativeness
            'metric': 'euclidean'
        }



class GUI(GUI_Settings):

    def __init__(
            self,
            *,
            polygon_filename: str = None,

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
        self.border_thickness = 5 #polygon border

        #First plots
        self.embed_band_list = [5,6,7]
        self.img_band_list = [5,6,7]

        #Other settings
        self.sample_size = sample_size
        self.random_state = random_state

        # Cache for border image to avoid redrawing
        self._cached_border_image = None
        self._cached_border_bands = None

        #Init tasks
        self.load_image()
        self.load_polygon()  # Load mask immediately (from file or generate from RGB)



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
    
    
    def get_all_band_names(self):
        '''
        Get all band names from the image header
        '''
        n_bands = self.image_dat.shape[2] if len(self.image_dat.shape) > 2 else 1
        return [self.image.band_name(i+1) for i in range(n_bands)]



    def generate_mask_from_rgb(self):
        '''
        Generate mask using dominant band logic with currently selected RGB bands.
        Uses the first band from img_band_list to compare against the other two.
        '''
        from dominant_band import dominant_band
        
        # Get the three RGB bands
        if len(self.img_band_list) < 3:
            print("Warning: Need at least 3 bands selected for mask generation")
            return None
        
        # Extract just the three RGB bands
        rgb_bands = self.img_band_list[:3]
        rgb_data = self.image_dat[..., [b-1 for b in rgb_bands]]
        
        # Use first band (R) as the dominant band to compare
        # band_index=1 means compare the first channel against others
        mask = dominant_band(X=rgb_data, band_index=1)
        
        return mask
    


    def load_polygon(self):
        '''
        Load or generate polygon mask.
        If polygon file provided, use it. Otherwise generate from RGB bands.
        '''
        self.polygon_dat = None
        self.mask_from_file = False

        if self.polygon_filename is not None:
            # Mask provided as file
            polygon = Raster(file_name=self.polygon_filename)

            if not polygon.is_polygon():
                raise ValueError(f"Not a polygon @ {self.polygon_filename}")
            
            self.polygon = polygon
            self.polygon_dat = polygon.read_bands('all').squeeze().astype(np.bool_)
            self.mask_from_file = True
            print("Using mask from file")
        else:
            # Generate mask from RGB bands
            print("Generating mask from RGB bands using 'dominant band' method")
            self.polygon_dat = self.generate_mask_from_rgb()
            self.mask_from_file = False

        #extract border from the rasterized polygon.
        self.border = extract_border(
            mask = self.polygon_dat, 
            thickness = self.border_thickness
        )

        #Just a guess of actual burn ratio
        self.guessed_burn_p = np.nanmean( self.polygon_dat )
    


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
    

    
    def get_border_image(self, band_list=None):
        '''
        Get image with border overlay, using cache to avoid repeated border drawing
        '''
        capped_band_list = self.img_band_list[:3] if band_list is None else band_list[:3]
        
        # Check if we can use cached version
        if (self._cached_border_image is not None and 
            self._cached_border_bands == capped_band_list):
            return self._cached_border_image
        
        # Generate new border image
        img_title, image = self.get_shown_image(band_list)
        border_image = draw_border(image, self.border)
        
        # Cache it
        self._cached_border_image = border_image
        self._cached_border_bands = capped_band_list
        
        return border_image


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

        t1 = time.time()

        print(f'Forest mapping done, cost {t1 - t0:.2f} s')

        cluster, _ = hdbscan_fit(
            self.current_embed, 
            **self.hdbscan_params
        )

        img_cluster, _ = hdbscan_approximate(
            transformed_img,
            cluster
        )

        print(f'Unique clusters: {np.unique(img_cluster)}')

        print(f'HDBSCAN done, cost {time.time() - t1:.3f}s')

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



    def save_classification(
            self,
            classification
    ):
        
        os.makedirs(self.save_dir, exist_ok=True)

        base_filename = os.path.basename(self.image_filename)

        writeENVI(
            output_filename=f'{self.save_dir}/{base_filename}_classified.bin',
            data = classification,
            mode='new',
            ref_filename=self.image_filename,
            band_names=['burned(bool)']
        )

        print('classification saved (ENVI).')



    def main(self):
        
        import matplotlib.patches as mpatches
        from matplotlib.widgets import Button

        self.sample_data() 

        embed_title, embed = self.get_band_embed()

        img_title, image = self.get_shown_image()
        
        # Get all band names from header
        all_band_names = self.get_all_band_names()

        ####### PLOT

        fig = plt.figure(figsize=(24, 12))
        
        # Create grid: left pane for pickers, right pane for plots
        # 3 columns total: [pickers | TSNE | Image]
        gs = GridSpec(20, 6, figure=fig, left=0.02, right=0.98, 
                     bottom=0.05, top=0.95, wspace=0.3, hspace=0.4)
        
        # Left pane - Band pickers (column 0)
        ax_embed_picker = fig.add_subplot(gs[0:8, 0])
        ax_embed_btn = fig.add_subplot(gs[8, 0])
        
        ax_img_picker = fig.add_subplot(gs[10:18, 0])
        ax_img_btn = fig.add_subplot(gs[18, 0])
        
        # Classify button (spanning middle columns)
        ax_classify_btn = fig.add_subplot(gs[19, 2:4])
        
        # Main plot areas - TSNE and Image side by side (columns 1-5)
        ax_tsne = fig.add_subplot(gs[0:19, 1:3])
        ax_img = fig.add_subplot(gs[0:19, 3:6])

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


        # Use cached border image
        img_plot = ax_img.imshow(self.get_border_image())
        
        ax_img.set_title(img_title)

        marker, = ax_img.plot([], [], "ro", markersize=6, fillstyle="none")

        # ----------------------------
        # Click logic for image
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

        '''
        Band List Pickers
        '''
        # Create band pickers (convert 1-based band indices to 0-based for picker)
        embed_picker = BandListPicker(
            ax_embed_picker,
            all_band_names,
            [b-1 for b in self.embed_band_list],
            title="TSNE Bands"
        )
        
        img_picker = BandListPicker(
            ax_img_picker,
            all_band_names,
            [b-1 for b in self.img_band_list],
            title="Image Bands (RGB)"
        )
        
        # Recalculate buttons
        def on_recalc_embed(event):
            new_band_list = embed_picker.get_selected_band_indices()
            
            if len(new_band_list) == 0:
                print("Please select at least one band for TSNE")
                return
            
            print(f"Updating TSNE bands to: {new_band_list}")
            self.embed_band_list = new_band_list

            # Recalculate embedding
            embed_title, embed = self.get_band_embed()

            # Update scatter plot
            tsne_plot.set_offsets(embed)
            ax_tsne.set_title(embed_title)

            fig.canvas.draw_idle()
        
        def on_recalc_img(event):
            new_band_list = img_picker.get_selected_band_indices()
            
            if len(new_band_list) != 3:
                # Show popup message
                popup_fig = plt.figure(figsize=(4, 2))
                popup_ax = popup_fig.add_subplot(111)
                popup_ax.axis('off')
                popup_ax.text(0.5, 0.5, 
                             f"Please select exactly 3 bands\n(Currently selected: {len(new_band_list)})",
                             ha='center', va='center', fontsize=12, weight='bold')
                plt.show()
                return
            
            print(f"Updating image bands to: {new_band_list}")
            self.img_band_list = new_band_list
            
            # Regenerate mask if it wasn't from a file
            if not self.mask_from_file:
                print("Regenerating mask from new RGB bands...")
                old_polygon = self.polygon_dat
                self.polygon_dat = self.generate_mask_from_rgb()
                
                # Update border
                self.border = extract_border(
                    mask = self.polygon_dat, 
                    thickness = self.border_thickness
                )
                
                # Update burn ratio guess
                self.guessed_burn_p = np.nanmean(self.polygon_dat)
                
                # Need to resample if polygon changed
                if old_polygon is not None:
                    self.sample_in_polygon = (self.polygon_dat.ravel())[self.sample_indices].astype(np.bool_)
                    
                    # Update scatter plot colors
                    tsne_colors = np.where(self.sample_in_polygon == 1, "red", "blue")
                    tsne_plot.set_color(tsne_colors)
            
            # Clear cache since bands changed
            self._cached_border_image = None

            img_title, image = self.get_shown_image()

            # Use cached border drawing
            img_plot.set_data(self.get_border_image())

            ax_img.set_title(img_title)

            fig.canvas.draw_idle()
        
        embed_recalc_btn = Button(ax_embed_btn, "Recalculate TSNE", 
                                   color='lightgreen', hovercolor='green')
        embed_recalc_btn.on_clicked(on_recalc_embed)
        
        img_recalc_btn = Button(ax_img_btn, "Update Image", 
                                color='lightgreen', hovercolor='green')
        img_recalc_btn.on_clicked(on_recalc_img)

        '''
        Classify Button with border
        '''
        classify_btn = Button(ax_classify_btn, "Classify", 
                             color='lightblue', hovercolor='skyblue')

        def on_submit_mapping(event):

            print(f'RUNNING ... Mapping on band list: {self.embed_band_list}')

            #Get clusters
            img_cluster = self.map_burn()

            img_cluster = img_cluster.reshape(self.image._ySize, self.image._xSize)

            #Get true classification
            classification = self.classify_cluster(img_cluster)

            #Plot product.
            fig2 = plt.figure(figsize=(12, 12))

            gs = GridSpec(
                2, 1,
                height_ratios=[12, 1],   # image big, button small
                figure=fig2
            )

            ax_img = fig2.add_subplot(gs[0])
    
            ax_img.imshow(
                classification, cmap = 'gray'
            )

            ax_img.set_title(f'Burn Mapping on {len(self.embed_band_list)} bands\n{self.current_band_name}')

            ax_img.set_xticks([])
            ax_img.set_yticks([])

            ########## Save classification button #########
            ax_btn = fig2.add_subplot(gs[1])
            ax_btn.set_xticks([])
            ax_btn.set_yticks([])
            for spine in ax_btn.spines.values():
                spine.set_visible(False)

            self.__save_clf_btn = Button(ax_btn, 'Save Classification',
                                         color='lightgreen', hovercolor='green')

            def save_clf(event):

                #Save to ENVI file.
                self.save_classification(classification)
            
            self.__save_clf_btn.on_clicked(save_clf)

            print(f'DONE! Mapping on band list: {self.embed_band_list}')

            plt.show()

        classify_btn.on_clicked(on_submit_mapping)

        plt.show()



if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("""python3 mapping_v2.py [raster input file (.bin)] [raster binary mask file (.bin)]
            [raster input file (.bin)] -- input raster file, ENVI format ( .bin and .hdr)  
            [raster mask file (.bin)] -- input raster mask file, 0/1 values, (.bin and .hdr) same dimensions as raster input file. This is an initial guess that guides the classification of the input file. This could be a rasterization of a polygon created by "heads-up" digitization, or a classification result generated by the "red wins" rule 
        
            Example:
            python3 mapping.py 1009.bin polygon_0000.bin""")
        sys.exit(1)

    image_filename = sys.argv[1]

    polygon_filename = sys.argv[2] if len(sys.argv) > 2 else None

    agent = GUI(
        polygon_filename=polygon_filename,
        image_filename=image_filename
    )
    
    # Start the main GUI
    agent.main()


