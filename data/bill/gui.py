'''
interactive_in_out.py (parallel dimensionality reduction)

version2: utilizes HPC
'''

########### LIBRARIES ##################

from raster import Raster

from misc.general import (
    htrim_3d,
    extract_border,
    draw_border
)

from sampling import (
    in_out_sampling,
    regular_sampling
)

import numpy as np

import sys

import ast


class GUI():

    def __init__(
            self,
            *,
            polygon_filename: str,
            image_filename: str,
            sample_size = 10000,
            random_state = 123
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

        #Other settings
        self.sample_size = sample_size
        self.random_state = random_state

        #Init tasks
        self.load_image()
        self.load_polygon()
        self.sample_data()# self.sample_in_out()


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
    



    def sample_data(
            self
    ):
        '''
        For visualization of embedding space, sampling is essential.

        main_raster is the focused raster.
        '''

        self.original_indices, self.samples = regular_sampling(
            raster_dat=self.image_dat,
            sample_size=self.sample_size,
            seed=self.random_state
        )



    def sample_in_out(
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
    


    def get_band_embed(
            self,
            X = None
    ):
        '''
        Uses TSNE algorithm to downsize n dim to just 2D

        Returns
        -------
        2D Embeddings
        '''

        from dim_reduce import tsne

        if X is None:
            X = self.samples

        embed = tsne(
            X, band_list=self.band_list
        )

        return embed
    

    def get_band_image(
            self
    ):
        
        return self.image_dat[..., [b-1 for b in self.band_list]]
    

    
    def get_shown_image(
            self,
            band_list = None
    ):
        '''
        Always use 'single' as key.

        Remember: band [1,2,3] looks different from [2,1,3].

        Band list item of 0 will be assumed to be 0.
        '''

        capped_band_list = self.band_list[:3] if band_list is None else band_list[:3]

        img_title = '  '.join(self.get_band_name(self.band_list))

        if (len(self.band_list) > 3):

            img_title += f" | Shows first 3 bands only."

        return img_title, htrim_3d( self.image_dat[..., [b - 1 for b in capped_band_list]] )
        


    def run(
            self
    ):
        '''
        Runs GUI
        '''
        import matplotlib.pyplot as plt

        self.band_list = [1,2,3]

        embed = self.get_band_embed()

        img_title, image = self.get_shown_image()

        #######

        fig, (ax_tsne, ax_img) = plt.subplots(1, 2, figsize=(20, 12))

        ax_tsne.axis("off")
        ax_img.axis("off")

        sc = ax_tsne.scatter(
            embed[:, 0], 
            embed[:, 1],
            s = 5,
            picker=3  # ← this enables clicking
        )

        ax_tsne.set_title(f"T-sne embedding | Sample Size: {self.sample_size} | Seed: {self.random_state}")

        #Right side of the plot, the main image (let parameter determine different band combination)

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

            flat = self.original_indices[k]

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


        # add textbox
        ax_box = fig.add_axes([0.35, 0.94, 0.1, 0.03])
        textbox = TextBox(ax_box, "Band list..e.g [1,2,3]: ")

        def on_submit(txt):

            self.band_list = ast.literal_eval(txt)

            try:
                #Set TNSE
                embed = self.get_band_embed()
                title, image = self.get_shown_image()

            except Exception:

                raise KeyError("This band combination is not in cache.")

            img_plot.set_data(
                draw_border(
                    image,
                    self.border
                )
            )

            ax_img.set_title(title)

            # ---- update scatter IN PLACE ----
            sc.set_offsets( embed )

            fig.canvas.draw_idle()

        textbox.on_submit(on_submit)
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

    