'''
interactive_in_out.py (parallel dimensionality reduction)
'''

########### LIBRARIES ##################

from raster import Raster

from misc.general import (
    htrim_3d,
    extract_border,
    draw_border
)

from sampling import in_out_sampling

from dim_reduce import (
    parDimRed
)

import numpy as np

import os

import sys

import ast


class GUI_settings:

    def __init__(
            self
    ):
        self.random_state = 123,
        self.in_sample_size = 300



class GUI(GUI_settings):

    def __init__(
            self,
            polygon_filename:str,
            image_filename: str
    ):
        '''
        Initialized parameters
        ----------------------
        in_sample_size: number of points to be sampled from inside the polygon.

        *out_sample_size: will automatically calculate using true ratio of in-out.
        '''

        super().__init__()
        
        #Default values
        self.polygon_filename = polygon_filename
        self.image_filename = image_filename

        #Init tasks
        self.load_image()
        self.load_polygon()
        self.load_dictionary()


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
            border_thickness: int = 7
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

        #extract border
        self.border = extract_border(
            mask=polygon.read_bands('all').squeeze(), 
            thickness=border_thickness
        )
    


    def sampling_in_out(
            self
    ):
        '''
        For visualization of embedding space, sampling is essential.

        main_raster is the focused raster.
        '''

        self.original_indices, self.samples, out_in_ratio = in_out_sampling(
            raster_dat=self.image_dat,
            polygon_dat=self.polygon.read_bands('all'),
            in_sample_size = self.in_sample_size,
            seed=self.random_state
        )

        self.out_sample_size = int( self.in_sample_size * out_in_ratio )
    


    def load_cache(
            self
    ):
        '''
        If data is already cached, load it.
        '''
        from joblib import load

        print("Loading Embeddings...")
        embed_dict =  load(self.CACHE_EMBEDDING_PATH)

        return embed_dict

    

    def generate_embedding(
            self
    ):
        '''
        Only used if data not cached.
        '''
        from misc.general import get_combinations
        from joblib import dump

        sams = self.samples

        band_combinations = get_combinations(
            val_lst=list(range(1, sams.shape[-1] + 1)),
            least=2
        )

        #Prepare tasks
        tasks = [
            (
                b, 
                sams[..., [bb - 1 for bb in b]]
            )
            
            for b in band_combinations
        ]

        embed_dict = parDimRed(tasks)

        print("Saving Embedding...")
        dump(embed_dict, self.CACHE_EMBEDDING_PATH)

        return embed_dict
            


    def load_dictionary(
            self
    ):
        '''
        If user changes embedding method or type, load cache or create if not cached.

        Only use when changing type and method.

        Do not use if changing band list.
        '''
        
        self.sampling_in_out()

        self.CACHE_EMBEDDING_PATH = f'caching/timestamp={self.image.acquisition_timestamp}&size={self.in_sample_size}&state={self.random_state}.joblib'

        if os.path.exists(self.CACHE_EMBEDDING_PATH):

            self.band_dictionary = self.load_cache()

        else:

            self.band_dictionary = self.generate_embedding()



    def get_band_embed(
            self,
            band_list: list
    ):
        '''
        This is used after a dictionary is loaded.

        UPDATE: 

        band_list = [1,2,4] is the same as [1,4,2]. Saved keys are in increasing order.
        '''

        sorted_band_list = sorted(band_list)

        try:
            embedding = self.band_dictionary[str(sorted_band_list)]

            return np.array(embedding)
        
        except Exception:
            
            raise KeyError("This band combination is not in KEYs.")
        

    
    def get_band_image(
            self,
            band_list: list
    ):
        '''
        Always use 'single' as key.

        Remember: band [1,2,3] looks different from [2,1,3].

        Band list item of 0 will be assumed to be 0.
        '''

        capped_band_list = band_list[:3]

        img_title = '  '.join(self.get_band_name(band_list))

        if (len(band_list) > 3):

            img_title += f" | Shows first 3 bands only."

        return img_title, self.image_dat[..., [b - 1 for b in capped_band_list]]
        


    def run(
            self
    ):
        '''
        Runs GUI
        '''
        import matplotlib.pyplot as plt

        band_list = [1,2,3]

        embed = self.get_band_embed(band_list)

        img_title, image = self.get_band_image(band_list)

        #######

        fig, (ax_tsne, ax_img) = plt.subplots(1, 2, figsize=(20, 8))

        ax_tsne.axis("off")
        ax_img.axis("off")

        sc_in = ax_tsne.scatter(
            embed[:self.in_sample_size, 0], 
            embed[:self.in_sample_size, 1],
            s = 20,
            c='red',
            label='Inside',
            picker=3  # ← this enables clicking
        )

        sc_out = ax_tsne.scatter(
            embed[self.in_sample_size:, 0], 
            embed[self.in_sample_size:, 1],
            s = 20,
            c='blue',
            label='Outside',
            picker=3
        )

        ax_tsne.set_title(f"Method: TSNE | In/Out sample:{self.in_sample_size} / {self.out_sample_size} | Seed: {self.random_state}")

        ax_tsne.legend()

        #Right side of the plot, the main image (let parameter determine different band combination)

        img_plot = ax_img.imshow(

            draw_border(
                htrim_3d( image ), #Because band convention starts at 1, but index is from 0
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

            if (event.artist is sc_in):
                flat = self.original_indices[:self.in_sample_size][k]

            elif (event.artist is sc_out):
                flat = self.original_indices[self.in_sample_size:][k]

            r = flat // W
            c = flat %  W

            hline.set_ydata([r, r])
            vline.set_xdata([c, c])

            hline.set_visible(True)
            vline.set_visible(True)

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("pick_event", on_pick)

        from matplotlib.widgets import TextBox

        fig.subplots_adjust(top = 2)

        # add textbox
        ax_box = fig.add_axes([0.35, 0.9, 0.3, 0.04])
        textbox = TextBox(ax_box, "Band list..e.g [1,2,3]: ")

        def on_submit(txt):

            band_lst = ast.literal_eval(txt)

            try:
                #Set TNSE
                embed = self.get_band_embed(band_lst)
                title, image = self.get_band_image(band_lst)

            except Exception:

                raise KeyError("This band combination is not in cache.")

            img_plot.set_data(
                draw_border(
                    htrim_3d(image),
                    self.border
                )
            )

            ax_img.set_title(title)

            # ---- update scatter IN PLACE ----
            sc_in.set_offsets(embed[:self.in_sample_size])
            sc_out.set_offsets(embed[self.in_sample_size:])

            fig.canvas.draw_idle()

        textbox.on_submit(on_submit)

        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.show()



if __name__ == '__main__':

    ### SOME DEFAULT VALUE, CAN SET AS INPUT LATER #######

    ############### handling argv #######################


    if len(sys.argv) < 3:
        print("Needs 1 raster file and 1 polygon file")
        sys.exit(1)

    image_filename = sys.argv[1]

    polygon_filename = sys.argv[2]

    agent = GUI(
        polygon_filename=polygon_filename,
        image_filename=image_filename
    )

    agent.run()


    