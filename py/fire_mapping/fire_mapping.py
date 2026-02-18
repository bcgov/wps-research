'''20260218 version
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
    Items are tightly packed just below the title.
    """
    def __init__(self, ax, band_names, initially_selected_indices, title="Select Bands"):
        self.ax = ax
        self.band_names = band_names
        self.selected_indices = set(initially_selected_indices)
        self.title = title
        
        # Visual properties
        self.font_size = 10
        self.bg_color_selected = 'lightblue'
        self.bg_color_unselected = 'white'
        self.text_color = 'black'
        
        n = len(band_names) if len(band_names) > 0 else 1
        
        # We need to figure out how tall each item should be in axes coords.
        # The axes has some pixel height; we want each item to be just tall
        # enough for the font. We use the DPI and font size to estimate.
        fig = self.ax.figure
        dpi = fig.dpi
        ax_bbox = self.ax.get_position()
        fig_h_in = fig.get_size_inches()[1]
        ax_h_px = ax_bbox.height * fig_h_in * dpi
        
        # Font height in pixels ≈ font_size * (dpi/72) + small padding
        font_h_px = self.font_size * (dpi / 72) + 4
        # Convert to axes fraction
        self.item_height = font_h_px / ax_h_px
        
        # Title uses same height as items (no extra gap)
        title_height = self.item_height
        
        # Setup axes
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        
        # Title at very top, flush
        self.title_obj = self.ax.text(
            0.5, 1.0, title,
            va='top', ha='center',
            fontsize=self.font_size, weight='bold')
        
        # First item starts immediately below title
        items_top = 1.0 - title_height
        
        # Create list items
        self.items = []
        for i, band_name in enumerate(band_names):
            y_top = items_top - i * self.item_height
            y_bottom = y_top - self.item_height
            
            is_selected = i in self.selected_indices
            rect = Rectangle(
                (0.0, y_bottom), 1.0, self.item_height,
                facecolor=self.bg_color_selected if is_selected else self.bg_color_unselected,
                edgecolor='gray', linewidth=0.5)
            self.ax.add_patch(rect)
            
            text = self.ax.text(
                0.03, y_bottom + self.item_height * 0.5,
                band_name,
                va='center', ha='left', fontsize=self.font_size,
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
        self.save_dir = '.'  # Save to current directory

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

        # Cache for border image to avoid redrawing
        self._cached_border_image = None
        self._cached_border_bands = None

        #Other settings
        self.sample_size = sample_size
        self.random_state = random_state

        #Init tasks - load image first
        self.load_image()
        
        # Set default bands based on image data
        # TSNE: all bands
        n_bands = self.image_dat.shape[2] if len(self.image_dat.shape) > 2 else 1
        self.embed_band_list = list(range(1, n_bands + 1))
        
        # RGB: intelligent search for B12, B11, B9
        self.img_band_list = self.find_default_rgb_bands()
        
        print(f"Default TSNE bands: all {n_bands} bands")
        print(f"Default RGB bands: {self.img_band_list} ({self.get_band_name(self.img_band_list)})")
        
        # Load polygon/mask
        self.load_polygon()



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
    
    
    def get_band_name_cleaned(
            self,
            band_list: list
    ):
        '''
        Get band names with common prefix removed for cleaner plot labels.
        
        E.g: ['1009.bin_001_B12_pre', '1009.bin_002_B11_pre'] 
             -> ['B12_pre', 'B11_pre'] (if '1009.bin_001_' is common prefix)
        '''
        band_names = self.get_band_name(band_list)
        
        if len(band_names) == 0:
            return band_names
        
        # Find common prefix
        if len(band_names) == 1:
            return band_names
        
        # Find shortest string to limit comparison
        min_len = min(len(name) for name in band_names)
        
        # Find common prefix length
        prefix_len = 0
        for i in range(min_len):
            if all(name[i] == band_names[0][i] for name in band_names):
                prefix_len = i + 1
            else:
                break
        
        # Remove common prefix if found
        if prefix_len > 0:
            cleaned = [name[prefix_len:] for name in band_names]
            return cleaned
        
        return band_names
    
    
    def get_all_band_names(self):
        '''
        Get all band names from the image header
        '''
        n_bands = self.image_dat.shape[2] if len(self.image_dat.shape) > 2 else 1
        return [self.image.band_name(i+1) for i in range(n_bands)]
    
    
    def find_default_rgb_bands(self):
        '''
        Find default RGB bands by searching for B12, B11, B9 patterns.
        
        Returns
        -------
        List of 3 band indices (1-based) in order [B12, B11, B9]
        
        Algorithm:
        1. Search all band names for B12, B11, B9 substrings
        2. Find groups where B12, B11, B9 appear consecutively (in order)
        3. Verify ordering is correct (indices should increase: B12 < B11 < B9)
        4. Use second group if multiple groups exist, otherwise first group
        '''
        all_band_names = self.get_all_band_names()
        
        # Find consecutive groups by scanning through bands
        groups = []
        
        i = 0
        while i < len(all_band_names):
            # Look for B12 starting at position i
            if 'B12' in all_band_names[i]:
                b12_idx = i + 1  # 1-based
                
                # Check if next 1-2 positions have B11
                b11_idx = None
                for j in range(i + 1, min(i + 3, len(all_band_names))):
                    if 'B11' in all_band_names[j]:
                        b11_idx = j + 1  # 1-based
                        
                        # Check if next 1-2 positions after B11 have B9
                        b9_idx = None
                        for k in range(j + 1, min(j + 3, len(all_band_names))):
                            if 'B9' in all_band_names[k]:
                                b9_idx = k + 1  # 1-based
                                
                                # Found a valid group!
                                # Verify ordering: B12 < B11 < B9
                                if b12_idx < b11_idx < b9_idx:
                                    groups.append([b12_idx, b11_idx, b9_idx])
                                    print(f"Found group: [{b12_idx}, {b11_idx}, {b9_idx}] = "
                                          f"[{all_band_names[b12_idx-1]}, {all_band_names[b11_idx-1]}, {all_band_names[b9_idx-1]}]")
                                else:
                                    raise ValueError(
                                        f"Bands with B12, B11, B9 substrings need to be ordered "
                                        f"highest-wavelength first (indices increasing). "
                                        f"Found ordering: "
                                        f"{all_band_names[b12_idx-1]} (idx {b12_idx}), "
                                        f"{all_band_names[b11_idx-1]} (idx {b11_idx}), "
                                        f"{all_band_names[b9_idx-1]} (idx {b9_idx})"
                                    )
                                break
                        break
            i += 1
        
        if len(groups) == 0:
            print("Warning: Could not find B12, B11, B9 pattern. Using bands 1, 2, 3 as default.")
            # Use first 3 bands as fallback
            n_bands = self.image_dat.shape[2] if len(self.image_dat.shape) > 2 else 3
            return [1, 2, 3] if n_bands >= 3 else [1, 1, 1]
        elif len(groups) == 1:
            print(f"Found 1 group with B12, B11, B9. Using: {groups[0]}")
            return groups[0]
        else:
            print(f"Found {len(groups)} groups with B12, B11, B9. Using second group: {groups[1]}")
            return groups[1]
    
    




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

        # Use cleaned band names (common prefix removed) - stored for classification title
        self.current_band_name = ' | '.join(self.get_band_name_cleaned(self.embed_band_list))

        # Simplified title without band listing (visible in picker)
        embed_title = f"T-SNE Embedding\nSample Size: {self.sample_size} | Seed: {self.random_state}"

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

        # Simplified title without band listing (visible in picker)
        img_title = "Image Bands RGB False-color"

        if (len(self.img_band_list) > 3):
            img_title += " | Warning! Showing first 3 bands only."

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

        output_filename = f'{self.save_dir}/{base_filename}_classified.bin'
        
        # Get absolute path for display
        abs_output_path = os.path.abspath(output_filename)
        
        writeENVI(
            output_filename=output_filename,
            data = classification,
            mode='new',
            ref_filename=self.image_filename,
            band_names=['burned(bool)']
        )

        print(f'Classification saved to: {abs_output_path}')
        print(f'  Header file: {abs_output_path.replace(".bin", ".hdr")}')



    def main(self):
        
        import matplotlib.patches as mpatches
        from matplotlib.widgets import Button

        self.sample_data() 

        embed_title, embed = self.get_band_embed()

        img_title, image = self.get_shown_image()
        
        # Get all band names from header
        all_band_names = self.get_all_band_names()

        ####### PLOT

        # ============================================================
        # Compute figure dimensions so content fills exactly to bottom.
        # ============================================================
        fig_width = 28  # inches
        
        picker_width_frac = 0.08
        button_height_frac = 0.035
        button_plot_gap = 0.002  # tiny gap so buttons don't overlap plots
        
        # Three columns share the non-picker width
        plot_width_frac = (1.0 - picker_width_frac) / 3.0
        plot_width_in = plot_width_frac * fig_width
        
        # Image aspect ratio determines plot height in inches
        img_h, img_w = self.image_dat.shape[:2]
        plot_height_in = plot_width_in * (img_h / img_w)
        
        # Total figure height: button + gap + plot, where
        # button_height_frac and gap are fractions of figure height.
        # plot_height_in = plot_height_frac * fig_height_in
        # plot_height_frac = 1.0 - button_height_frac - button_plot_gap
        # So: fig_height_in = plot_height_in / (1.0 - button_height_frac - button_plot_gap)
        content_frac = 1.0 - button_height_frac - button_plot_gap
        fig_height = plot_height_in / content_frac
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        
        # ============================================================
        # LAYOUT — zero gaps everywhere.
        # ============================================================
        
        # Buttons: top edge at 1.0
        button_bottom = 1.0 - button_height_frac
        
        # Plots: fill from just below buttons to bottom of figure
        plot_top = button_bottom - button_plot_gap
        plot_bottom = 0.0
        plot_height = plot_top - plot_bottom
        
        # Band pickers: sized to exactly fit content, flush together
        # Compute item height in figure-fraction using same formula as BandListPicker
        n_bands = len(all_band_names)
        dpi = fig.dpi
        font_size = 10  # must match BandListPicker.font_size
        font_h_px = font_size * (dpi / 72) + 4
        fig_h_px = fig_height * dpi
        item_height_fig = font_h_px / fig_h_px  # one item in figure fraction
        
        # Each picker needs: 1 title + n items = (n+1) * item_height
        picker_content_height = (n_bands + 1) * item_height_fig
        
        # Top picker: top edge at 1.0 (flush with window top, aligned with buttons)
        embed_picker_top = 1.0
        embed_picker_bottom = embed_picker_top - picker_content_height
        
        # Bottom picker: immediately below top picker, no gap
        img_picker_top = embed_picker_bottom
        img_picker_bottom = img_picker_top - picker_content_height
        
        ax_embed_picker = fig.add_axes([0.0, embed_picker_bottom, picker_width_frac, picker_content_height])
        ax_img_picker   = fig.add_axes([0.0, img_picker_bottom,   picker_width_frac, picker_content_height])
        
        # Column 1 — TSNE
        tsne_left = picker_width_frac
        ax_embed_btn = fig.add_axes([tsne_left, button_bottom, plot_width_frac, button_height_frac])
        ax_tsne      = fig.add_axes([tsne_left, plot_bottom,   plot_width_frac, plot_height])
        
        # Column 2 — Image
        img_left = tsne_left + plot_width_frac
        ax_img_btn = fig.add_axes([img_left, button_bottom, plot_width_frac, button_height_frac])
        ax_img     = fig.add_axes([img_left, plot_bottom,   plot_width_frac, plot_height])
        
        # Column 3 — Classification
        class_left = img_left + plot_width_frac
        ax_classify_btn   = fig.add_axes([class_left, button_bottom, plot_width_frac, button_height_frac])
        ax_classification = fig.add_axes([class_left, plot_bottom,   plot_width_frac, plot_height])

        # --- Remove all axes chrome ---
        for ax in [ax_tsne, ax_img, ax_classification]:
            ax.axis("off")
            ax.set_xmargin(0)
            ax.set_ymargin(0)

        tsne_colors = np.where(self.sample_in_polygon == 1, "red", "blue")

        tsne_plot = ax_tsne.scatter(
            embed[:, 0], 
            embed[:, 1],
            s = 5,
            c = tsne_colors,
            picker = 3 #enables clicking
        )

        # Title INSIDE the axes (overlaid on plot content)
        ax_tsne.text(
            0.5, 0.99, embed_title,
            transform=ax_tsne.transAxes,
            va='top', ha='center', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
        )

        # Legend overlaid INSIDE the TSNE plot
        leg = ax_tsne.legend(
            handles = [
            mpatches.Patch(color="blue", label="Outside"),
            mpatches.Patch(color="red",  label="Inside"),
        ], loc='upper right', fontsize=7, framealpha=0.8,
           borderpad=0.3, handlelength=1.0, handletextpad=0.3,
           borderaxespad=0.3)
        # Ensure legend stays inside axes and doesn't expand the axes
        leg.set_in_layout(False)


        # Use cached border image — aspect='auto' fills pre-sized axes perfectly
        img_plot = ax_img.imshow(self.get_border_image(), aspect='auto')
        
        # Title INSIDE the axes (overlaid on image)
        ax_img.text(
            0.5, 0.99, img_title,
            transform=ax_img.transAxes,
            va='top', ha='center', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
        )

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
            ax_tsne.autoscale_view()
            # Clear old title text and re-add inside axes
            for txt in ax_tsne.texts:
                txt.remove()
            ax_tsne.text(
                0.5, 0.99, embed_title,
                transform=ax_tsne.transAxes,
                va='top', ha='center', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
            )

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

            # Clear old title text and re-add inside axes
            for txt in ax_img.texts:
                txt.remove()
            ax_img.text(
                0.5, 0.99, img_title,
                transform=ax_img.transAxes,
                va='top', ha='center', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
            )

            fig.canvas.draw_idle()
        
        embed_recalc_btn = Button(ax_embed_btn, "Recalculate TSNE", 
                                   color='lightgreen', hovercolor='green')
        embed_recalc_btn.on_clicked(on_recalc_embed)
        
        img_recalc_btn = Button(ax_img_btn, "Update Image", 
                                color='lightgreen', hovercolor='green')
        img_recalc_btn.on_clicked(on_recalc_img)

        '''
        Classify Button and Classification Display
        '''
        classify_btn = Button(ax_classify_btn, "Classify", 
                             color='lightblue', hovercolor='skyblue')
        
        # Initialize classification display
        ax_classification.axis("off")
        classification_plot = None
        
        # Store latest classification
        self.latest_classification = None

        def on_submit_mapping(event):

            print(f'RUNNING ... Mapping on band list: {self.embed_band_list}')

            #Get clusters
            img_cluster = self.map_burn()

            img_cluster = img_cluster.reshape(self.image._ySize, self.image._xSize)

            #Get true classification
            classification = self.classify_cluster(img_cluster)
            
            # Store for potential future use
            self.latest_classification = classification

            # Auto-save classification immediately
            self.save_classification(classification)

            # Display in third pane
            ax_classification.clear()
            ax_classification.imshow(classification, cmap='gray', aspect='auto')
            ax_classification.text(
                0.5, 0.99, 'Classification',
                transform=ax_classification.transAxes,
                va='top', ha='center', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
            )
            ax_classification.set_xticks([])
            ax_classification.set_yticks([])
            ax_classification.axis("off")

            print(f'DONE! Mapping on band list: {self.embed_band_list}')

            fig.canvas.draw_idle()

        classify_btn.on_clicked(on_submit_mapping)

        # Auto-run classification on startup
        print("Auto-running classification on startup...")
        on_submit_mapping(None)

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
