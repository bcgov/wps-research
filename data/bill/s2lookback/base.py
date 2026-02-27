
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
    mask_dir: str = None
    output_dir: str = 's2lookback_temp'

    #Basic settings
    mask_threshold: float = 0.01
    min_mask_prop_trigger: str = 0.0
    max_mask_prop_usable: str = 0.8
    max_lookback_days: int = 7
    nodata_val = np.nan

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

    #Parallel processing
    n_workers: int = 8

    #htrim hardcoded
    lo = np.array([1028., 1048., 1067.])
    hi = np.array([3049., 3638., 2290.])



    def get_file_dictionary(
            self
    ):
        
        from s2lookback.utils import get_ordered_file_dict

        file_dict = get_ordered_file_dict(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            start=self.start,
            end=self.end
        )

        if len(file_dict) == 0:

            raise ValueError('\nEmpty satisfying files, cannot process\nCheck start and end dates inputs.')
        
        return file_dict
    


    def get_file_path(
            self,
            date: datetime,
            key: str
    ):
        
        return self.file_dict[date][key]
    


    def read_image(
            self, 
            date: datetime,
            band_list: list[int] = [1,2,3]
    ):
        
        '''
        Read data from image.
        Combines multiple images if image_path is a list.
        '''
        image_paths = self.get_file_path(date, 'image_path')

        if band_list is None: band_list = 'all'

        if isinstance(image_paths, str):

            return Raster(image_paths).read_bands(band_list)

        return [Raster(img_f).read_bands(band_list) for img_f in image_paths]
    
    

    def read_mask(
            self,
            date: datetime,
            as_prob: bool = False
    ):
        '''
        Read mask on that date.
        Combines multiple masks if mask_path is a list.
        Also returns coverage of the mask.

        If as_prob=True:
            - Returns list of arrays if multiple masks, single array if one mask
            - Binary masks (0/1) are returned as bool even with as_prob=True
            - Probabilistic masks are returned as float in [0, 1]
            - Coverage is returned as None when returning a list
        If as_prob=False:
            - All masks are combined via logical OR into a single bool mask
            - Returns (mask, coverage)
        '''
        mask_paths = self.get_file_path(date, 'mask_path')
        single     = isinstance(mask_paths, str)
        if single:
            mask_paths = [mask_paths]

        processed = []
        for mask_f in mask_paths:
            raw = Raster(mask_f).read_bands().squeeze()

            max_val = raw.max()

            if max_val <= 1.0:
                # Already in [0, 1] or binary — no scaling needed
                processed.append(raw.astype(np.float32))
            else:
                # Assumed to be in [0, 100] — scale down
                processed.append((raw / 100.).astype(np.float32))

        if as_prob:
            if single:
                return processed[0], float(processed[0].mean())
            return processed, None

        # Combine into a single boolean mask
        bool_masks = []
        for data in processed:
            if self.mask_threshold is None:
                bool_masks.append(data.astype(np.bool_))
            else:
                bool_masks.append(data >= self.mask_threshold)

        mask     = np.logical_or.reduce(bool_masks)
        coverage = float(mask.mean())
        return mask, coverage
        


    def get_nodata_mask(
            self,
            img_dat: np.ndarray,
            nodata_val: None = None
    ):

        if (nodata_val is None):
            nodata_val = self.nodata_val
        
        return np.all(
            img_dat == nodata_val, 
            axis=-1
        )



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
        

    

