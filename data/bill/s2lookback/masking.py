'''
A look back method.

This module attempts to mask cloud, cloud shadow.

1. Prepare training data.

2. 
'''

from s2lookback.base import LookBack

import os

import joblib

from pathlib import Path

import numpy as np

from datetime import datetime

from dataclasses import dataclass

from cuml.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from cuml.model_selection import train_test_split

@dataclass
class MASK(LookBack):

    def __post_init__(self):
        self.validate_method()



    def validate_method(
            self
    ):
        '''
        Checks for all conditions before running.
        '''
        if self.sample_size is None:

            raise ValueError('This method requires a sample size, received None.')
        


    def prepare_samples_single(
            self, 
            cur_date: datetime,
            prev_date: datetime
    ):
        '''
        Helper for prepare samples
        '''

        img_dat_main = self.read_image(cur_date)
        nodata_cur = np.all(img_dat_main == 0, axis=-1)

        img_dat_prev = self.read_image(prev_date)
        nodata_prev = np.all(img_dat_prev == 0, axis=-1)

        mask_cur, _ = self.read_mask(cur_date)
        
        ALL_NODATA_MASK = np.logical_or(nodata_prev, nodata_cur)

        #Data Engineering

        DIFF = (img_dat_main - img_dat_prev) / (img_dat_main + img_dat_prev + 1e-3)

        IMG_DAT = np.dstack([DIFF, img_dat_main / 10000, img_dat_prev / 10000])

        VALID_IMG_DAT = IMG_DAT[~ALL_NODATA_MASK]

        VALID_MASK_CUR = mask_cur[~ALL_NODATA_MASK]

        #Filter no data out, we dont want to use this.

        return self.sample_datasets(VALID_IMG_DAT, VALID_MASK_CUR)
    


    def prepare_samples(
            self,
            cur_date: datetime,
            previous_dates_list: list[datetime]
    ):
        all_X = None
        all_y = None

        for i, prev_date in enumerate(previous_dates_list):

            print(f'>>> Co-sampling with {prev_date} ({i+1} before)')

            X, y = self.prepare_samples_single(
                cur_date, prev_date
            )

            if all_X is None: 
                all_X = X
                all_y = y

            else:
                all_X = np.vstack([all_X, X])
                all_y = np.concatenate([all_y, y])
        
        return all_X, all_y
        



    def train_and_report(
            self,
            X, y
    ):
        '''
        Use classification models.
        '''
        print('\n>>> Training model...')

        #Prepare train and test data.
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y,
                                                            test_size=0.2, shuffle=True, random_state=42)

        pd = {'max_depth': 15, 'n_estimators': 200}

        rf = RandomForestClassifier(**pd)

        rf.fit(X_train, y_train)

        predictions = rf.predict(X_test)

        print(classification_report(y_test, predictions))

        return rf



    def save_model(
            self, 
            model, date
    ):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        print("<<< Saving model.")

        model_path = Path(self.output_dir) / f'model_{date}.joblib'

        joblib.dump(model, model_path)



    def run(
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

            print('-' * 50)

            #If true, it means mrap is necessary
            print(f"{i+1}. Getting ready for {cur_date} ...")
            
            previous_dates_list = get_dates_within(
                datetime_list=date_list[:i][::-1], 
                current_datetime=cur_date, 
                N_days=self.max_lookback_days
            )

            if len(previous_dates_list) == 0:

                print("Skipped.., no previous dates.")

                continue

            all_X, all_y = self.prepare_samples(
                cur_date,
                previous_dates_list
            )

            model = self.train_and_report(all_X, all_y)

            self.save_model(model, cur_date)



if __name__ == "__main__":

    masker = MASK(
        image_dir='C11659/L1C/resampled_20m',
        mask_dir='C11659/cloud_20m',
        output_dir='C11659/cloud_models',
        sample_size=100_000,
        mask_threshold=0.001,
        start = datetime(2025, 8, 1),
        end = datetime(2025, 8, 30)
    )

    masker.run()



        

        








        
        
        
        
    

    

