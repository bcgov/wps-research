'''
Most recent available pixels (MRAP)
'''
import os

from pathlib import Path

import numpy as np

from raster import Raster

from misc import writeENVI

from datetime import datetime


class MRAP:

    def __init__(
                self,
                #Directory paths
                image_dir: str,
                cloud_dir: str,

                *,

                #General
                cloud_threshold: float = 0.01,
                min_cloud_prop_trigger: float = 0.01,
                max_mrap_days: int = 9,
                
                #For Ajusting only
                adjust_lighting: bool = False,
                sample_prop: float = 0.1,

                #Generate png
                png: bool = False
        ):

        self.image_dir = image_dir
        self.cloud_dir = cloud_dir

        self.cloud_threshold = cloud_threshold
        self.min_cloud_prop_trigger = min_cloud_prop_trigger
        self.max_mrap_days = max_mrap_days

        self.adjust_lighting = adjust_lighting
        self.sample_prop = sample_prop

        self.mrap_dir = 'C11659/L1C/mrap'

        self.png = png



    def get_file_dictionary(
            self
    ):
        
        from s2mrap.utils import get_ordered_file_dict

        self.file_dict = get_ordered_file_dict(image_dir=self.image_dir,
                                               cloud_dir=self.cloud_dir)
        


    def read_image(
            self,
            date
    ):
        '''
        Read data from image.

        Buffering: Check if image will be used again, and buffer. Otherwise, remove from buffer
        '''
        img_f = self.file_dict[date]['image_path']

        img_dat = Raster(img_f).read_bands()

        return img_dat
    


    def read_cloud(
            self,
            date: datetime
    ):
        '''
        Read cloud on that date. 
        
        Scale to [0, 1] if necessary.

        Buffering: Check if cloud will be used again, and buffer. Otherwise, remove from buffer
        '''
        cloud_f = self.file_dict[date]['cloud_path']

        cloud_prob = Raster(cloud_f).read_bands().squeeze() / 100.

        return cloud_prob



    def get_cloud_mask(
            self,
            cloud_prob: np.ndarray
    ):
        '''
        Thresholds cloud probability and returns cloud mask.

        Pixel with probablity less than threshold will be considered non-cloud.
        '''

        mask = (cloud_prob >= self.cloud_threshold)

        coverage = mask.mean()

        return mask, coverage
    


    def fill_engine(
            self,
            image_dat_main: np.ndarray,
            cloud_mask_main: np.ndarray,
            previous_dates_list: list[datetime]
    ):
        '''
        Main heart of MRAP.

        Fill cloud pixels with pixels from the previous dates

        * Current version: No data in main will not be filled. 

        * Cloudy pixel will not be filled with previous 'no data'.
        '''
        if len(previous_dates_list) < 1:

            return image_dat_main, cloud_mask_main.mean()

        for prev_date in previous_dates_list:
            
            #1. Prepare data and mask for mrap
            image_dat_prev = self.read_image(prev_date)
            nodata_prev = np.all(image_dat_prev == 0, axis=-1)

            cloud_prob_prev = self.read_cloud(prev_date)
            cloud_mask_prev, _ = self.get_cloud_mask(cloud_prob_prev)

            #2. Mask where it is cloudy in main but not cloudy and not 'no data' in prev
            fill_mask = np.logical_and(
                cloud_mask_main, 
                np.logical_and(~cloud_mask_prev, ~nodata_prev)
            )

            #3. Fill
            image_dat_main[fill_mask] = image_dat_prev[fill_mask]

            #3. After filling, mask has changed as well (pixels that used to be cloudy but not filled will still be cloudy)
            cloud_mask_main = np.logical_and(cloud_mask_main, ~fill_mask)

            current_coverage = cloud_mask_main.mean()

            if current_coverage < self.min_cloud_prop_trigger:

                break

        return image_dat_main, current_coverage
        


    def fill(
            self
    ):
        '''
        1. Match files in dictionary
        '''

        self.get_file_dictionary()
        date_list = list(self.file_dict.keys())

        '''
        2. Iterate from the 2nd date to the final date. 

            We do not mrap on first date, because there was no data prior to the 1st date (dates were sorted).
        '''
        from s2mrap.utils import get_dates_within

        if not os.path.exists(self.mrap_dir):
            os.mkdir(self.mrap_dir)

        for i, cur_date in enumerate(date_list):

            #1. Check if this date should be mrapped
            cloud_prob_main = self.read_cloud(cur_date)
            image_dat_main = self.read_image(cur_date)

            cloud_mask_main, coverage = self.get_cloud_mask(cloud_prob_main)

            print('-' * 50)

            if coverage >= self.min_cloud_prop_trigger:

                #If true, it means mrap is necessary
                print(f"{i+1}. Filling {cur_date} ...")
                
                previous_dates_list = get_dates_within(
                    datetime_list=date_list[:i][::-1], 
                    current_datetime=cur_date, 
                    N_days=self.max_mrap_days
                )

                image_dat_main, post_fill_coverage = self.fill_engine(
                    image_dat_main,
                    cloud_mask_main,
                    previous_dates_list
                )

                print(f"Completed, coverage changed from {coverage:.3f} to {post_fill_coverage:.3f}")

            else:
                print(f"{i+1}. Skipped.\n>> Current cloud coverage {coverage:.3f} is less than preset {self.min_cloud_prop_trigger}")


            #Finally, write file.
            image_path = Path(self.file_dict[cur_date]['image_path'])

            writeENVI(
                output_filename=f'{self.mrap_dir}/{image_path.stem}_MRAP.bin',
                data = image_dat_main,
                ref_filename = image_path,
                mode = 'new',
                same_hdr = True
            )

        if (self.png):

            from photos import save_png_same_dir
            
            print('' * 50)
            print('Saving png... may take some time')

            save_png_same_dir(
                dir = self.mrap_dir
            )


if __name__ == "__main__":

    mrap = MRAP(
        image_dir='C11659/L1C/resampled',
        cloud_dir='C11659/cloud_20m',
        png = True
    )

    mrap.fill()






            




