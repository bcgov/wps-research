'''
A look back method.

Most recent available pixels (MRAP)
'''
from s2lookback.base import LookBack

import os

from pathlib import Path

import numpy as np

from datetime import datetime

from misc import writeENVI

from dataclasses import dataclass


@dataclass
class MRAP(LookBack):

    # adjust_lighting: bool = False


    def fill_engine(
            self,
            image_dat_main: np.ndarray,
            mask_dat_main: np.ndarray,
            previous_dates_list: list[datetime]
    ):
        '''
        Main heart of MRAP.

        Fill masked pixels with pixels from the previous dates

        * Current version: No data in main will not be filled. 

        * Masked pixel will not be filled with previous 'no data'.
        '''
        if len(previous_dates_list) < 1:

            return image_dat_main, mask_dat_main.mean()

        for prev_date in previous_dates_list:
            
            #1. Prepare data and mask for mrap
            image_dat_prev = self.read_image(prev_date)
            nodata_prev = np.all(image_dat_prev == 0, axis=-1)

            mask_dat_prev, _ = self.read_mask(prev_date)

            #2. Mask where it is mask in main but not mask and not 'no data' in prev
            fill_mask = np.logical_and(
                mask_dat_main, 
                np.logical_and(~mask_dat_prev, ~nodata_prev)
            )

            #3. Fill
            image_dat_main[fill_mask] = image_dat_prev[fill_mask]

            #4. After filling, mask has changed as well (pixels that used to be masky but not filled will still be masky)
            mask_dat_main = np.logical_and(mask_dat_main, ~fill_mask)

            current_coverage = mask_dat_main.mean()

            if current_coverage < self.min_mask_prop_trigger:

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
        from s2lookback.utils import get_dates_within

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        for i, cur_date in enumerate(date_list):

            #1. Check if this date should be mrapped
            mask_dat_main, coverage = self.read_mask(cur_date)

            image_dat_main = self.read_image(cur_date)

            print('-' * 50)

            if coverage >= self.min_mask_prop_trigger:

                #If true, it means mrap is necessary
                print(f"{i+1}. Filling {cur_date} ...")
                
                previous_dates_list = get_dates_within(
                    datetime_list=date_list[:i][::-1], 
                    current_datetime=cur_date, 
                    N_days=self.max_lookback_days
                )

                image_dat_main, post_fill_coverage = self.fill_engine(
                    image_dat_main,
                    mask_dat_main,
                    previous_dates_list
                )

                print(f"Completed, coverage changed from {coverage:.3f} to {post_fill_coverage:.3f}")

            else:
                print(f"{i+1}. Skipped.\n>> Current mask coverage {coverage:.3f} is less than preset {self.min_mask_prop_trigger}")


            #Finally, write file.
            image_path = Path(self.file_dict[cur_date]['image_path'])

            writeENVI(
                output_filename=f'{self.output_dir}/{image_path.stem}_MRAP.bin',
                data = image_dat_main,
                ref_filename = image_path,
                mode = 'new',
                same_hdr = True
            )

        if (self.png):

            self.save_png()



if __name__ == "__main__":

    mrap = MRAP(
        image_dir='C11659/L1C/resampled_20m',
        mask_dir='C11659/cloud_20m',
        mask_threshold=0.001, #at least 0.1% to be considered cloud
        output_dir='C11659/mrap',
        start=datetime(2025, 9, 10),
        end=datetime(2025, 10, 1)
    )

    mrap.fill()






            




