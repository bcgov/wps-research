
'''
Base class for Look Back methods.
'''

from dataclasses import dataclass, field

from raster import Raster

import numpy as np

from datetime import datetime


@dataclass
class LookBack:

    #Files
    image_dir: str
    mask_dir: str
    output_dir: str

    #Basic settings
    mask_threshold: float = None
    max_lookback_days: int = 7

    #Date selections
    start: datetime = None
    end: datetime = None

    #Random sampling using mask
    sample_size: int = None
    sample_between_prop: dict[str, float]= field(default_factory=lambda: {'mask': 0.7, 'non-mask': 0.3})
    sample_within_prop: dict[str, float] = field(default_factory=lambda: {'mask': 0.75, 'non-mask': 0.75})

    #Miscellaneous
    png: bool = True


    def get_file_dictionary(
            self
    ):
        
        from s2lookback.utils import get_ordered_file_dict

        self.file_dict = get_ordered_file_dict(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            start=self.start,
            end=self.end
        )

        if len(self.file_dict) == 0:

            raise ValueError('\nEmpty satisfying files, cannot process\nCheck start and end dates.')
        


    def read_image(
            self,
            date
    ):
        '''
        Read data from image.
        '''
        img_f = self.file_dict[date]['image_path']

        img_dat = Raster(img_f).read_bands()

        return img_dat
    


    def read_mask(
            self,
            date: datetime
    ):
        '''
        Read mask on that date.

        Also returns coverage of the mask
        '''
        mask_f = self.file_dict[date]['mask_path']

        mask_prob = Raster(mask_f).read_bands().squeeze() / 100.

        if (self.mask_threshold is None):
            mask = mask_prob.astype(np.bool_)

        else:
            mask = (mask_prob >= self.mask_threshold)

        coverage = mask.mean()

        return mask, coverage
    


    def sample(
            self
    ):
        pass
        


    def save_png(
            self
    ):
        
        from photos import save_png_same_dir
            
        print('' * 50)
        print('Saving png... may take some time')

        save_png_same_dir(
            dir = self.output_dir
        )
        

    

