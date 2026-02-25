'''
A look back method.

This module attempts to reduce haze.
'''

from s2lookback.base import LookBack

import os

import joblib

from pathlib import Path

import numpy as np

from datetime import datetime

from dataclasses import dataclass, field

from cuml.ensemble import RandomForestRegressor

from misc import writeENVI

from matplotlib.figure import Figure

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



@dataclass
class HAZELESS(LookBack):

    revert_within_pct: float = 0.2

    reg_n_iter: int = 3

    rf_reg_params: dict = field(default_factory=lambda: {
        'n_estimators': 80,    
        'max_depth': 12,
        'max_features': "sqrt"
    })



    def __post_init__(self):
        
        self.validate_method()

        os.makedirs(self.output_dir, exist_ok=True)

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

            for b in range( y.shape[-1] ):
                
                #Data prep#############

                l, h = np.percentile(
                    diff[:, b], (30, 70)
                )

                diff_mask = (diff[:, b] > l) & (diff[:, b] < h)

                #Randomize indices
                idx = np.random.choice(diff_mask.sum(), 
                                    size=int(diff_mask.sum() * 0.005), 
                                    replace=False)

                X_tr = X[diff_mask, b][idx].reshape(-1, 1)
                y_tr = y[diff_mask, b][idx]
                ########################

                #Model fitting#############

                reg = RandomForestRegressor(
                    **self.rf_reg_params
                )
                
                reg.fit(X_tr, y_tr)
                ############################


                #Predict and Record###########
                predictions = reg.predict(X[:, b].reshape(-1, 1))

                prediction_record[:, b, it] = predictions
                #######################

            print(f' >> Done #{it + 1}.')

        return np.median(prediction_record, axis = 2)
    


    def revert(
            self,
            enhanced_dat: np.ndarray,
            mrap_dat: np.ndarray,
            mask: np.ndarray
    ):
        '''
        If hazeless data is no different from mrap within n% across all bands (preset), revert to mrap.
        '''

        reverted_enhanced_dat = enhanced_dat.copy()

        A = mrap_dat[mask]

        B = enhanced_dat[mask]

        rel_diff = np.abs(A - B) / (np.abs(A) + 1e-6)

        revert_mask = np.all(rel_diff <= 0.2, axis=1)

        cloud_mask_flat = mask.ravel()

        full_revert_mask = np.zeros_like(cloud_mask_flat, dtype=bool)

        full_revert_mask[cloud_mask_flat] = revert_mask

        reverted_enhanced_dat.reshape(-1, reverted_enhanced_dat.shape[-1])[full_revert_mask] = \
                mrap_dat.reshape(-1, mrap_dat.shape[-1])[full_revert_mask]
        

        return reverted_enhanced_dat



    def hazeless(
            self,
            date: datetime
    ):
        
        #Get data
        img_dat_ls = self.read_image(date)
        raw_dat, mrap_dat = img_dat_ls[0], img_dat_ls[1]
        mask_dat, _ = self.read_mask(date)

        #No data in raw
        nodata_mask = self.get_nodata_mask(raw_dat)

        #Data differencing (within the masked data)
        masked_raw, masked_mrap = raw_dat[mask_dat], mrap_dat[mask_dat]
        masked_diff = masked_raw - masked_mrap

        #Prep
        X = masked_raw.reshape(-1, raw_dat.shape[-1])
        y = masked_mrap.reshape(-1, raw_dat.shape[-1])

        hazeless_dat = self.regress_record(
            masked_diff, X, y
        )
        
        #Copy raw for hazeless map
        enhanced_dat = raw_dat.copy()

        enhanced_dat[mask_dat] = hazeless_dat

        #Revert for better quality
        reverted_enhanced_dat = self.revert(
            enhanced_dat, 
            mrap_dat,
            mask_dat 
        )

        #Final step, brought everything in no_data from mrap to raw (no change)

        reverted_enhanced_dat[nodata_mask] = mrap_dat[nodata_mask]

        #Save figs

        fig = Figure(figsize=(21, 8))
        canvas = FigureCanvas(fig)
        axes = fig.subplots(1, 3)

        axes[0].imshow((raw_dat[..., :3] - self.lo) / (self.hi - self.lo))
        axes[0].set_title("Raw")
        axes[0].axis("off")

        axes[1].imshow((mrap_dat[..., :3] - self.lo) / (self.hi - self.lo))
        axes[1].set_title("MRAP")
        axes[1].axis("off")

        axes[2].imshow((reverted_enhanced_dat[..., :3] - self.lo) / (self.hi - self.lo))
        axes[2].set_title("Reverted Enhanced")
        axes[2].axis("off")

        fig.tight_layout()

        #Save fig
        fig.savefig(Path(self.output_dir) / f"{date}.png")



    def run(self):
        
        '''
        1. Match files in dictionary
        '''

        date_list = list(self.file_dict.keys())

        for i, date in enumerate(date_list):
            
            print(f'{i+1}. Enhancing {date} ...')

            self.hazeless(date)



if __name__ == "__main__":

    hazeless = HAZELESS(
        image_dir=['C11659/L1C/resampled_20m', 'C11659/wps_cloud/adjusted_mrap'],
        mask_dir = 'C11659/cloud_20m',
        output_dir= 'C11659/wps_cloud/hazeless_mrap',
        start=datetime(2025, 8, 30)
    )

    hazeless.run()



        

        








        
        
        
        
    

    

