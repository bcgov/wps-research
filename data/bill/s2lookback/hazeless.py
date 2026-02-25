# '''
# A look back method.

# This module attempts to reduce haze.
# '''

# from s2lookback.base import LookBack

# import os

# from pathlib import Path

# import numpy as np

# from datetime import datetime

# from dataclasses import dataclass

# from sklearn.linear_model import LinearRegression

# from misc import writeENVI

# from matplotlib.figure import Figure

# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



# @dataclass
# class HAZELESS(LookBack):

#     revert_all_within_pct: float = 0.2

#     revert_swir_exceed_pct: float = 0.2

#     reg_n_iter: int = 5


#     def __post_init__(self):
        
#         self.validate_method()

#         os.makedirs(self.output_dir, exist_ok=True)

#         self.file_dict = self.get_file_dictionary()



#     def validate_method(self):

#         if not (len(self.image_dir) == 2):

#             raise ValueError("Needs Raw and MRAP.")



#     def regress_record(
#             self,
#             diff: np.ndarray,
#             X: np.ndarray,
#             y: np.ndarray,
#     ):
        
#         prediction_record = np.zeros((X.shape[0], X.shape[1], self.reg_n_iter))
        
#         for it in range(self.reg_n_iter):

#             for b in range( y.shape[-1] ):
                
#                 #Data prep#############

#                 l, h = np.percentile(
#                     diff[:, b], (30, 70)
#                 )

#                 diff_mask = (diff[:, b] > l) & (diff[:, b] < h)

#                 #Randomize indices
#                 idx = np.random.choice(diff_mask.sum(), 
#                                     size=min(50_000, int(diff_mask.sum() * 0.005)), 
#                                     replace=False)

#                 X_tr = X[diff_mask][idx]
#                 y_tr = y[diff_mask, b][idx]
#                 ########################

#                 #Model fitting#############
#                 reg = LinearRegression()
                
#                 reg.fit(X_tr, y_tr)
#                 ############################


#                 #Predict and Record###########
#                 predictions = reg.predict(X[:])

#                 prediction_record[:, b, it] = predictions
#                 #######################

#         return np.median(prediction_record, axis = 2)
    


#     def revert(
#             self,
#             raw_dat: np.ndarray,
#             enhanced_dat: np.ndarray,
#             mrap_dat: np.ndarray,
#             mask: np.ndarray
#     ):
#         '''
#         If hazeless data is no different from mrap within n% across all bands (preset), revert to mrap.
#         '''

#         print(" > Reverting to mrap ...")
#         reverted_enhanced_dat = enhanced_dat.copy()

#         #For RAW diff
#         A = raw_dat[mask]

#         B = enhanced_dat[mask]

#         rel_diff = np.abs(A - B) / (np.abs(A) + 1e-6)

#         revert_mask_raw = np.all(rel_diff[..., :2] >= self.revert_swir_exceed_pct, axis=1)

#         #For MRAP diff

#         A = mrap_dat[mask]

#         rel_diff = np.abs(A - B) / (np.abs(A) + 1e-6)

#         revert_mask_mrap = np.all(rel_diff <= self.revert_all_within_pct, axis=1)

#         #Create full revert mask

#         cloud_mask_flat = mask.ravel()

#         full_revert_mask = np.zeros_like(cloud_mask_flat, dtype=bool)

#         full_revert_mask[cloud_mask_flat] = revert_mask_mrap | revert_mask_raw

#         reverted_enhanced_dat.reshape(-1, reverted_enhanced_dat.shape[-1])[full_revert_mask] = \
#                 mrap_dat.reshape(-1, mrap_dat.shape[-1])[full_revert_mask]
        
#         return reverted_enhanced_dat



#     def hazeless(
#             self,
#             date: datetime
#     ):
        
#         #Get data
#         img_dat_ls = self.read_image(date)
#         raw_dat, mrap_dat = img_dat_ls[0], img_dat_ls[1]
#         mask_dat, _ = self.read_mask(date)

#         #No data in raw
#         nodata_mask = self.get_nodata_mask(raw_dat)

#         #Data differencing (within the masked data)
#         masked_raw, masked_mrap = raw_dat[mask_dat], mrap_dat[mask_dat]
#         masked_diff = masked_raw - masked_mrap

#         #Prep
#         X = masked_raw.reshape(-1, raw_dat.shape[-1])
#         y = masked_mrap.reshape(-1, raw_dat.shape[-1])

#         hazeless_dat = self.regress_record(
#             masked_diff, X, y
#         )
        
#         #Copy raw for hazeless map
#         enhanced_dat = raw_dat.copy()

#         enhanced_dat[mask_dat] = hazeless_dat

#         #Revert for better quality
#         reverted_enhanced_dat = self.revert(
#             raw_dat,
#             enhanced_dat, 
#             mrap_dat,
#             mask_dat 
#         )

#         #Final step, brought everything in no_data from mrap to raw (no change)

#         reverted_enhanced_dat[nodata_mask] = mrap_dat[nodata_mask]

#         #Save figs

#         fig = Figure(figsize=(21, 8))
#         canvas = FigureCanvas(fig)
#         axes = fig.subplots(1, 3)

#         axes[0].imshow((raw_dat[..., :3] - self.lo) / (self.hi - self.lo))
#         axes[0].set_title("Raw")
#         axes[0].axis("off")

#         axes[1].imshow((enhanced_dat[..., :3] - self.lo) / (self.hi - self.lo))
#         axes[1].set_title("Enhanced")
#         axes[1].axis("off")

#         axes[2].imshow((reverted_enhanced_dat[..., :3] - self.lo) / (self.hi - self.lo))
#         axes[2].set_title("Reverted Enhanced")
#         axes[2].axis("off")

#         fig.tight_layout()

#         #Save fig
#         fig.savefig(Path(self.output_dir) / f"{date}.png")

#         return reverted_enhanced_dat



#     def run(self):
        
#         '''
#         1. Match files in dictionary
#         '''

#         date_list = list(self.file_dict.keys())

#         for i, date in enumerate(date_list):
            
#             print(f'{i+1}. Enhancing {date} ...')

#             final_dat = self.hazeless(date)

#             #Write Data

#             image_path = self.file_dict[date]['image_path'][1]

#             writeENVI(
#                 output_filename=Path(self.output_dir) / f'{Path(image_path).stem}_HAZELESS.bin',
#                 data = final_dat,
#                 ref_filename=image_path,
#                 same_hdr=True
#             )



# if __name__ == "__main__":

#     hazeless = HAZELESS(
#         image_dir=['C11659/L1C/resampled_20m', 'C11659/wps_cloud/adjusted_mrap'],
#         mask_dir = ['C11659/cloud_20m', 'C11659/wps_shadow'],
#         output_dir= 'C11659/wps_cloud/hazeless_mrap_all_bands',
#         start=datetime(2025, 8, 30),
#         end=datetime(2025,9, 2)
#     )

#     hazeless.run()



'''
A look back method.

This module attempts to reduce haze.
'''

from s2lookback.base import LookBack

import os

from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path

import numpy as np

from datetime import datetime

from dataclasses import dataclass, field

from sklearn.linear_model import LinearRegression

from misc import writeENVI

from matplotlib.figure import Figure

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



# ---------------------------------------------------------------------------
# Module-level worker — must be at top level to be picklable
# ---------------------------------------------------------------------------

def _process_date(args: dict):
    """
    Reconstructs a HAZELESS instance in the child process and runs one date.
    Returns (index, date, success, message).
    """
    idx         = args['idx']
    date        = args['date']
    init_kwargs = args['init_kwargs']
    file_entry  = args['file_entry']

    try:
        instance = HAZELESS(**init_kwargs)

        # Overwrite with single entry — skips full rebuild in __post_init__
        instance.file_dict = {date: file_entry}

        final_dat = instance.hazeless(date)

        image_path = file_entry['image_path'][1]

        writeENVI(
            output_filename=Path(init_kwargs['output_dir']) / f'{Path(image_path).stem}_HAZELESS.bin',
            data=final_dat,
            ref_filename=image_path,
            same_hdr=True
        )

        return idx, date, True, 'OK'

    except Exception as e:
        return idx, date, False, str(e)



@dataclass
class HAZELESS(LookBack):

    revert_all_within_pct:  float = 0.2
    revert_swir_exceed_pct: float = 0.2
    reg_n_iter:             int   = 5
    n_workers:              int   = 4
    _prebuilt_file_dict:    dict  = field(default=None, repr=False)


    def __post_init__(self):

        self.validate_method()

        os.makedirs(self.output_dir, exist_ok=True)

        # If a pre-built dict was passed in (worker reconstruction), use it directly
        if self._prebuilt_file_dict is not None:
            self.file_dict = self._prebuilt_file_dict
        elif not getattr(self, 'file_dict', None):
            self.file_dict = self.get_file_dictionary()


    def validate_method(self):

        if not (len(self.image_dir) == 2):
            raise ValueError("Needs Raw and MRAP.")


    def regress_record(
            self,
            diff: np.ndarray,
            X: np.ndarray,
            y: np.ndarray,
    ):

        prediction_record = np.zeros((X.shape[0], X.shape[1], self.reg_n_iter))

        for it in range(self.reg_n_iter):

            for b in range(y.shape[-1]):

                # Data prep
                l, h = np.percentile(diff[:, b], (30, 70))

                diff_mask = (diff[:, b] > l) & (diff[:, b] < h)

                # Randomize indices
                idx = np.random.choice(
                    diff_mask.sum(),
                    size=min(50_000, int(diff_mask.sum() * 0.005)),
                    replace=False
                )

                X_tr = X[diff_mask][idx]
                y_tr = y[diff_mask, b][idx]

                # Model fitting
                reg = LinearRegression()
                reg.fit(X_tr, y_tr)

                # Predict and record
                prediction_record[:, b, it] = reg.predict(X[:])

        return np.median(prediction_record, axis=2)


    def revert(
            self,
            raw_dat: np.ndarray,
            enhanced_dat: np.ndarray,
            mrap_dat: np.ndarray,
            mask: np.ndarray
    ):
        '''
        If hazeless data is no different from mrap within n% across all bands (preset), revert to mrap.
        '''
        reverted_enhanced_dat = enhanced_dat.copy()

        # For RAW diff
        A = raw_dat[mask]
        B = enhanced_dat[mask]

        rel_diff = np.abs(A - B) / (np.abs(A) + 1e-6)

        revert_mask_raw = np.all(rel_diff[..., :2] >= self.revert_swir_exceed_pct, axis=1)

        # For MRAP diff
        A = mrap_dat[mask]

        rel_diff = np.abs(A - B) / (np.abs(A) + 1e-6)

        revert_mask_mrap = np.all(rel_diff <= self.revert_all_within_pct, axis=1)

        # Create full revert mask
        cloud_mask_flat = mask.ravel()

        full_revert_mask = np.zeros_like(cloud_mask_flat, dtype=bool)

        full_revert_mask[cloud_mask_flat] = revert_mask_mrap | revert_mask_raw

        reverted_enhanced_dat.reshape(-1, reverted_enhanced_dat.shape[-1])[full_revert_mask] = \
            mrap_dat.reshape(-1, mrap_dat.shape[-1])[full_revert_mask]

        return reverted_enhanced_dat


    def hazeless(
            self,
            date: datetime
    ):

        # Get data
        img_dat_ls = self.read_image(date)
        raw_dat, mrap_dat = img_dat_ls[0], img_dat_ls[1]
        mask_dat, _ = self.read_mask(date)

        # No data in raw
        nodata_mask = self.get_nodata_mask(raw_dat)

        # Data differencing (within the masked data)
        masked_raw, masked_mrap = raw_dat[mask_dat], mrap_dat[mask_dat]
        masked_diff = masked_raw - masked_mrap

        # Prep
        X = masked_raw.reshape(-1, raw_dat.shape[-1])
        y = masked_mrap.reshape(-1, raw_dat.shape[-1])

        hazeless_dat = self.regress_record(masked_diff, X, y)

        # Copy raw for hazeless map
        enhanced_dat = raw_dat.copy()
        enhanced_dat[mask_dat] = hazeless_dat

        # Revert for better quality
        reverted_enhanced_dat = self.revert(
            raw_dat,
            enhanced_dat,
            mrap_dat,
            mask_dat
        )

        # Final step: fill no_data from mrap
        reverted_enhanced_dat[nodata_mask] = mrap_dat[nodata_mask]

        # Save figs
        fig = Figure(figsize=(21, 8))
        canvas = FigureCanvas(fig)
        axes = fig.subplots(1, 3)

        axes[0].imshow(np.clip((raw_dat[..., :3] - self.lo) / (self.hi - self.lo), 0, 1))
        axes[0].set_title("Raw")
        axes[0].axis("off")

        axes[1].imshow(np.clip((enhanced_dat[..., :3] - self.lo) / (self.hi - self.lo), 0, 1))
        axes[1].set_title("Enhanced")
        axes[1].axis("off")

        axes[2].imshow(np.clip((reverted_enhanced_dat[..., :3] - self.lo) / (self.hi - self.lo), 0, 1))
        axes[2].set_title("Reverted Enhanced")
        axes[2].axis("off")

        fig.tight_layout()
        fig.savefig(Path(self.output_dir) / f"{date}.png")

        return reverted_enhanced_dat


    def _build_init_kwargs(self) -> dict:
        return dict(
            image_dir              = self.image_dir,
            mask_dir               = self.mask_dir,
            output_dir             = self.output_dir,
            start                  = self.start,
            end                    = self.end,
            revert_all_within_pct  = self.revert_all_within_pct,
            revert_swir_exceed_pct = self.revert_swir_exceed_pct,
            reg_n_iter             = self.reg_n_iter,
            n_workers              = self.n_workers,
            _prebuilt_file_dict    = self.file_dict,   # pass already-built dict
        )


    def run(self):

        date_list   = list(self.file_dict.keys())
        init_kwargs = self._build_init_kwargs()

        job_args = [
            {
                'idx':         i,
                'date':        date,
                'init_kwargs': init_kwargs,
                'file_entry':  self.file_dict[date],
            }
            for i, date in enumerate(date_list)
        ]

        print(f"Processing {len(date_list)} dates with {self.n_workers} workers ...\n")

        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:

            futures = {pool.submit(_process_date, args): args for args in job_args}

            for future in as_completed(futures):
                idx, date, success, msg = future.result()
                status = '✓' if success else '✗'
                print(f'  [{idx + 1}/{len(date_list)}] {status} {date}  {msg if not success else ""}')



if __name__ == "__main__":

    hazeless = HAZELESS(
        image_dir  = ['C11659/L1C/resampled_20m', 'C11659/wps_cloud/adjusted_mrap'],
        mask_dir   = ['C11659/cloud_20m', 'C11659/wps_shadow'],
        output_dir = 'C11659/wps_cloud/hazeless_mrap',
        n_workers  = 16,
    )

    hazeless.run()


        