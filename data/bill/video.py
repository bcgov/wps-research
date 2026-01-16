'''
generates a gif to see how the environment changes from a certain date.
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


class GIF():
    
    def __init__(
            self,
            folder_name:str,
            ref_date: str
    ):
        '''
        ref day is the date to start tracking on.
        '''
        self.foldername = folder_name

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




    def __update_barc(
            self,
            new_data
    ):
        
        '''
        Plot 3 and 6
        '''

        #MEthod (choose between barc, change)
        _, nbr_post_1, dnbr = dNBR(
            NIR_1=self.ref_data[..., 3],
            SWIR_1=self.ref_data[..., 0],

            NIR_2=new_data[..., 3],
            SWIR_2=new_data[..., 0]
        )

        self.im2.set_data(
            dnbr
        )  

        #Method
        # _, nbr_post_2, dnbr_next_day = dNBR(
        #     NIR_1=self.prev_data[..., 3],
        #     SWIR_1=self.prev_data[..., 0],

        #     NIR_2=new_data[..., 3],
        #     SWIR_2=new_data[..., 0]
        # )


        #Difference between dNBR today and dNBR the day before

        nbr_post_2 = NBR(NIR = self.prev_data[..., 3],
                         SWIR= self.prev_data[..., 0])

        self.im6.set_data(
            (nbr_post_2 - nbr_post_1) / (nbr_post_2 + nbr_post_1 + 1e-3)
        )  


    def __update_change_det(
            self, 
            new_data
    ):
        '''
        Plot 3,4,5
        '''
        #Process data for plotting

        #MEthod (choose between barc, change)
        change = change_detection(
            pre_X=self.ref_data,
            post_X=new_data
        )

        self.im3.set_data(
            self.__raw2plot(change)
        )  


        #SWIR wins

        swir_wins = dominant_band(
            change,
            band_index=1
        )

        self.im4.set_data(
            swir_wins
        )  


        #NIR wins
        nir_wins = dominant_band(
            change,
            band_index=4
        )

        self.im5.set_data(
            nir_wins
        )  
    

    def __update(
        self, idx
    ):
        '''
        Use this as main function of updates
        '''
        
        date = self.date_list[idx]

        #Load Reference image, this will be fixed
        new_data = self.__get_raster_data(
            self.date_list[idx]
        )

        #Plot 1: the next date
        self.im1.set_data(
            self.__raw2plot(new_data)
        )

        self.title1.set_text(f"index: {idx} | date={date}")

        #Plot 2: barc
        self.__update_barc(new_data)

        #Plot 3,4,5: change detection
        self.__update_change_det(new_data)


        self.prev_data = new_data
        

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
            fps=1.5,                # control speed here
            codec="libx264",
            bitrate=2000
        )
        
        anim.save("videos/changes.mp4", writer=writer)



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
        import numpy as np


        self.fig = plt.figure(
            figsize=(20, 10),
            constrained_layout=True
        )

        gs = GridSpec(
            nrows=2,
            ncols=4,
            figure=self.fig,
            height_ratios=[1, 1],   # control row heights
            width_ratios=[1, 1, 1, 1]
        )

        # first row
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax2 = self.fig.add_subplot(gs[0, 1])
        ax3 = self.fig.add_subplot(gs[0, 2])
        ax4 = self.fig.add_subplot(gs[0, 3])

        # second row
        ax5 = self.fig.add_subplot(gs[1, 3])
        ax6 = self.fig.add_subplot(gs[1, 1])


        #Load Reference image, this will be fixed
        self.ref_data = self.__get_raster_data(
            self.date_list[0]
        )

        self.prev_data = self.ref_data

        temp = np.zeros((self.ref_data.shape[0], self.ref_data.shape[1]))

        #The middle data
        self.im1 = ax1.imshow(temp, vmin=0, vmax=1)
        self.title1 = ax1.set_title(f"date={self.date_list[0]}")

        #method 1 on 2 imageries
        self.im2 = ax2.imshow(temp, vmin=0, vmax=1, cmap = 'gray')
        self.title2 = ax2.set_title(f"dNBR")

        #method 2 on 2 imageries
        self.im3 = ax3.imshow(temp, vmin=0, vmax=1)
        self.title3 = ax3.set_title(f"Change Detection")

        #method 3 on 2 imageries
        self.im4 = ax4.imshow(temp, vmin=0, vmax=1, cmap = 'gray')
        self.title4 = ax4.set_title(f"Change Detection (SWIR wins)")

        self.im5 = ax5.imshow(temp, vmin=0, vmax=1, cmap = 'gray')
        self.title5 = ax5.set_title(f"Change Detection (NIR wins)")

        #dNBR by date
        self.im6 = ax6.imshow(temp, vmin=0, vmax=1, cmap = 'gray')
        self.title6 = ax6.set_title(f"dNBR (between every 2 adjacent images)")


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
        ref_date = ref_date
    )

    g.run()

    print(f"Generating took {time() - t0} s")

    