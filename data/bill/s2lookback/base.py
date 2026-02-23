
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
    mask_threshold: float = 1e-4
    min_mask_prop_trigger: str = 0.01
    max_lookback_days: int = 7

    #Date selections
    start: datetime = None
    end: datetime = None

    #Random sampling using mask
    sample_size: int = None
    sample_between_prop: dict[str, float] = field(default_factory=lambda: {'mask': 0.5, 'non_mask': 0.5})
    sample_within_prop: dict[str, float] = field(default_factory=lambda: {'mask': 0.8, 'non_mask': 0.5})
    mask_samp_sz = 0
    non_mask_samp_sz = 0

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

            raise ValueError('\nEmpty satisfying files, cannot process\nCheck start and end dates inputs.')
        


    def read_image(
            self,
            date: datetime
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
    


    def sample_datasets(
            self,
            img_dat: np.ndarray,
            mask: np.ndarray[np.bool_]
    ):
        '''
        Samples from the data.

        There are 2 labels only (binary).
        '''

        #Sample masked pixels
        d = img_dat[mask]
        size = int( min(
            self.sample_size * self.sample_between_prop['mask'], 
            d.shape[0] * self.sample_within_prop['mask']
        ) )

        sampled_idx = np.random.choice(
            d.shape[0], size, 
            replace=False
        )

        mask_samples = d[sampled_idx]
        mask_labels = np.full(size, 1)
        
        self.mask_samp_sz += size
        print(f"  + {size} samples of mask | Label = 1 | Total {self.mask_samp_sz}")


        #Sample non-masked pixels
        d = img_dat[~mask]
        size = int( min(
            self.sample_size * self.sample_between_prop['non_mask'], 
            d.shape[0] * self.sample_within_prop['non_mask']
        ) )

        sampled_idx = np.random.choice(
            d.shape[0], size, 
            replace=False
        )

        non_mask_samples = d[sampled_idx]
        non_mask_labels = np.full(size, 0)

        self.non_mask_samp_sz += size
        print(f"  + {size} samples of non_mask | Label = 0 | Total {self.non_mask_samp_sz}")

        #Concatenate date

        X = np.vstack([mask_samples, non_mask_samples])
        y = np.concatenate([mask_labels, non_mask_labels])
        
        return X, y
        


    def save_png(
            self
    ):
        
        from photos import save_png_same_dir
            
        print('' * 50)
        print('Saving png... may take some time')

        save_png_same_dir(
            dir = self.output_dir
        )
        

    

