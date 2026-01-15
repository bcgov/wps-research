

from exceptions.gui_exception import *

from misc.general import htrim_3d

from change_detection import change_detection



class GUI():
    
    def __init__(
            self,
            folder_name:str,
            ref_date: str
    ):
        '''
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
        from exceptions.gui_exception import Load_Data_Error


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

                date_str = date.isoformat()

                raster, _ = minimum_nan_raster(files)

                #raster_by_date_dict[date_str] = raster
                raster_by_date_dict[date_str] = raster.read_bands(crop=True)


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

        # data = raster.read_bands(crop = True)

        return raster
    


    def __raw2plot(
            self,
            raw
    ):
        '''
        '''
        return htrim_3d(
            raw[..., :3]
        )



    def __update_gui(
            self,
            i
    ):
        
        i = int(i)
        date = self.date_list[i]

        # ----------------------------
        # TODO: YOUR LOGIC GOES HERE
        # ----------------------------
        # Example idea:
        #   current = read_raster_for(date)        # (H,W) or (H,W,C)
        #   derived = your_burn_mapping(baseline, current)
        #
        # Then update plots:
        #   imM.set_data(current_display)
        #   imR.set_data(derived_display)
        #
        # Demo placeholders:

        #Load Reference image, this will be fixed
        new_data = self.__get_raster_data(
            self.date_list[i]
        )

        #Process data for plotting
        plotting_data = self.__raw2plot(new_data)

        self.imM.set_data(plotting_data)

        self.titleM.set_text(f"date={date}")

        #MEthod
        change = change_detection(
            pre_X=self.ref_data,
            post_X=new_data
        )

        self.imR.imshow(
            self.__raw2plot(change)
        )
    
        self.fig.canvas.draw_idle()



    def __gui(
            self
    ):
        '''
        Description
        -----------


        Parameters
        ----------
        date_lst: unique date list of ascending order

        referencing_date: The date taken as reference
        '''

        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button


        self.fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(14, 8))


        #Load Reference image, this will be fixed
        self.ref_data = self.__get_raster_data(
            self.date_list[0]
        )

        #Process data for plotting
        plotting_data = self.__raw2plot(self.ref_data)

        self.imL = axL.imshow(plotting_data)
        axL.set_title(f"Reference date: {self.date_list[0]}")

        self.imM = axM.imshow(plotting_data)
        self.titleM = axM.set_title(f"date={self.date_list[0]}")


        #The method on 2 imageries
        self.imR = axR

        self.titleR = axR.set_title(f"Change Detection")

        #Sliding
        ax_sl = self.fig.add_axes([0.12, 0.12, 0.6, 0.04])
        sl = Slider(ax_sl, "idx", 0, len(self.date_list) - 1, valinit=0, valstep=1)

        #Start from the next day
        sl.on_changed(self.__update_gui)
    

        plt.show()



    def run(
            self
    ):
        print("Preparing Data ...")
        self.__prepare_data()

        print("Checking for errors ...")
        self.__check()

        print("Showing...")
        self.__gui()
        

    

if __name__ == "__main__":

    ref_date = '20250522' #From fire_testing

    gui = GUI(
        folder_name='./fire_testing',
        ref_date = ref_date
    )

    gui.run()

    