'''
generates a VIDEO to see how the environment changes from a certain date.
'''


from exceptions.data import *

from misc.general import (
    htrim_1d,
    htrim_3d
)

from change_detection import change_detection

from dominant_band import dominant_band

from barc import (
    NBR,
    dNBR
)

from time import time

import os

import numpy as np

from bayesian_map import *



class GIF():
    
    def __init__(
            self,
            folder_name:str,
            video_filename: str,
            ref_date: str
    ):
        '''
        ref day is the date to start tracking on.
        '''
        self.foldername = folder_name

        self.video_filename = video_filename

        self.ref_date = ref_date



    def __prepare_data(
            self
    ):
        '''
        '''
        
        from misc.sen2 import get_date_dict
        from raster import (
            minimum_nan_raster
        )
        from exceptions.data import Load_Data_Error


        #Get the unique dates in ascending order, just dates
        unique_dates = get_date_dict(
            folder=self.foldername,
            from_date=self.ref_date
        )

        #Load the raster, can take some time to filter rasters
        #Can be memory heavy, for testing use a folder with just a few ENVI files.
        raster_by_date_dict = {}

        try:

            for date, files in unique_dates.items():

                #Corner of debate, should we store raster or its images??
                #Raster would allow you flexibility, while images would be faster (no need to read again), but you cant read another band

                date_str = date.strftime("%Y%m%d")

                raster, _ = minimum_nan_raster(files)

                raster_by_date_dict[date_str] = raster


        except Exception:

            raise Load_Data_Error("Cannot load raster data.")
        

        self.raster_by_date_dict = raster_by_date_dict
        
        self.date_list = list(raster_by_date_dict.keys())


        return



    def __check(
            self
    ):
        '''
        First round:

        Check for any problems before running...
        '''
    
        #Needs at least 2 files
        if ( len(self.date_list) < 2 ):

            raise Not_Enough_Information("Need at least 2 different dates to work.")


        #Reference data means which date to fix on the left plot, so that we can see after than.
        if (self.ref_date not in self.date_list):

            print(f'Warning, this date: {self.ref_date} is not in the folder.')

        
        return



    def __get_raster_data(
            self,
            date: str
    ):
        '''
        Get data based on date.

        This method extracts all bands
        '''
        raster = self.raster_by_date_dict[date]

        data = raster.read_bands(
            crop = True
        )

        return data
    


    def __raw2plot(
            self,
            raw
    ):
        '''
        From multiband data to just enough for visualization
        '''

        return htrim_3d(
            raw[..., :3]
        )



    def __im1(
            self
    ):
        '''
        Raw data of next date

        Caching: no need.
        '''

        self.im1.set_data(
            self.__raw2plot(self.current_data)
        )

        self.title1.set_text(self.current_date)


    def __im2(
            self
    ):
        '''
        dNBR
        '''

        _, _, dnbr = dNBR(
            NIR_1=self.ref_data[..., 3],
            SWIR_1=self.ref_data[..., 0],

            NIR_2=self.current_data[..., 3],
            SWIR_2=self.current_data[..., 0]
        )


        self.im2.set_data(
            dnbr
        )  

        return dnbr
    

    def __im3(
            self,
            dnbr
    ):
        '''
        Bayesian Mapping
        '''

        self.alpha, self.beta = bayesian_update_2(
            alpha=self.alpha,
            beta=self.beta,
            new_dnbr=dnbr,
        )

        expectations = beta_expectation(self.alpha, self.beta)

        # maps = make_prediction(expectations, 0.5)

        self.im3.set_data(expectations)

        return

    def __im4(
            self,
            dnbr
    ):
        
        self.im4.set_data(
            is_evidence(dnbr)
        )
        return


    

    def __update(
        self, idx
    ):
        '''
        Use this as main function of updates
        '''

        #Load Reference image, this will be fixed
        self.current_date = self.date_list[idx]

        self.current_data = self.__get_raster_data(
            date=self.current_date
        )

        #Change data shown in each subfig

        self.__im1()

        dnbr = self.__im2()

        self.__im3(dnbr)

        self.__im4(dnbr)

        #Just in case
        self.prev_date = self.current_date
        
        self.prev_data = self.current_data
        

        if idx % 5 == 0:

            print(f'Updated {idx} / {len(self.date_list)}')



    def __run_animation(
            self
    ):
        
        from matplotlib.animation import FuncAnimation, FFMpegWriter

        #Start from the next day
        anim = FuncAnimation(
            self.fig,
            self.__update,
            frames=range(1, len(self.date_list)),
            blit=False
        )

        writer = FFMpegWriter(
            fps=1,                # control speed here
            codec="libx264",
            bitrate=2000
        )
        
        anim.save(self.video_filename, writer=writer)



    def __layout_and_run(
            self
    ):
        '''
        Parameters
        ----------
        date_lst: unique date list of ascending order

        referencing_date: The date taken as reference
        '''

        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.colors import LinearSegmentedColormap
        import numpy as np

        cmap = LinearSegmentedColormap.from_list(
            "green_to_red",
            ["#b7e4c7", "#640002"]  # light green → red
        )


        self.fig = plt.figure(
            figsize=(20, 20),
            constrained_layout=True
        )

        gs = GridSpec(
            nrows=2,
            ncols=2,
            figure=self.fig,
            height_ratios=[1, 1],   # control row heights
            width_ratios=[1, 1]
        )

        # first row
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax2 = self.fig.add_subplot(gs[0, 1])

        # second row
        ax3 = self.fig.add_subplot(gs[1, 0])
        ax4 = self.fig.add_subplot(gs[1, 1])


        #Load Reference image, this will be fixed
        self.ref_date = self.date_list[0]

        self.ref_data = self.__get_raster_data(
            date = self.ref_date
        )

        self.prev_data = self.ref_data

        temp = np.zeros((self.ref_data.shape[0], self.ref_data.shape[1]))

        #The next date data
        self.im1 = ax1.imshow(temp, vmin=0, vmax=1)
        self.title1 = ax1.set_title(f"date={self.date_list[0]}")

        #dNBR
        self.im2 = ax2.imshow(temp, vmin=0, vmax=1, cmap='gray')
        self.title2 = ax2.set_title(f"dNBR")

        #Bayesian Mapping
        self.im3 = ax3.imshow(temp, vmin=0, vmax=1, cmap = cmap)
        self.title3 = ax3.set_title(f"Bayesian (redder -> higher prob of Burn)")

        self.alpha = np.ones((self.ref_data.shape[0], self.ref_data.shape[1]))
        self.beta = np.ones((self.ref_data.shape[0], self.ref_data.shape[1]))


        self.im4 = ax4.imshow(temp, vmin=0, vmax=1)
        self.title4 = ax4.set_title(f"scaled dNBR (>= 80)")

        #We dont need axisfor ax in self.fig.axes:
        for ax in self.fig.axes:
            ax.axis("off")


        #Run animation
        self.__run_animation()


        plt.close(self.fig)
        


    def run(
            self
    ):
        print("Preparing Data ...")
        self.__prepare_data()

        print("Checking for errors ...")
        self.__check()

        print("Generating video...")
        self.__layout_and_run()

        print('...Done, enjoy.')

    

if __name__ == "__main__":

    import sys

    from time import time


    if len(sys.argv) < 3:
        print("Needs a folder and a start date")
        sys.exit(1)

    folder = sys.argv[1]
    ref_date = sys.argv[2]


    t0 = time()

    g = GIF(
        folder_name=folder,
        video_filename='./videos/bayesian_4_days.mp4',
        ref_date = ref_date
    )

    g.run()

    print(f"Generating took {time() - t0} s")

    