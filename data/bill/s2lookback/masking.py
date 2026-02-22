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

        img_dat_cur = self.read_image(cur_date)
        nodata_cur = np.all(img_dat_cur == 0, axis=-1)

        img_dat_prev = self.read_image(prev_date)
        nodata_prev = np.all(img_dat_prev == 0, axis=-1)

        mask_cur, _ = self.read_mask(cur_date)
        
        ALL_NODATA_MASK = np.logical_or(nodata_prev, nodata_cur)

        #Data Engineering

        DIFF = (img_dat_cur - img_dat_prev) / (img_dat_cur + img_dat_prev + 1e-3)

        IMG_DAT = np.dstack([DIFF, img_dat_cur / 10_000, img_dat_prev / 10_000])

        VALID_IMG_DAT = IMG_DAT[~ALL_NODATA_MASK]

        VALID_MASK_CUR = mask_cur[~ALL_NODATA_MASK]

        #Filter no data out, we dont want to use this.

        return self.sample_datasets(VALID_IMG_DAT, VALID_MASK_CUR)
    


    def prepare_samples(
            self,
            cur_date: datetime,
            previous_dates_list: list[datetime]
    ):
        

        #Sample for train data
        prev_, cur_ = previous_dates_list[0], previous_dates_list[1]

        print(f'\n***\n > Sampling TRAIN data: d1 = {prev_} & d2 = {cur_}.')

        X_train, y_train = self.prepare_samples_single(
            cur_, prev_
        )

        if self.X_train is None: 
            self.X_train = X_train
            self.y_train = y_train

        else:
            self.X_train = np.vstack([self.X_train, X_train])
            self.y_train = np.concatenate([self.y_train, y_train])

        #Sample for test data
        print(f'\n***\n > Sampling TEST data: d1 = {previous_dates_list[1]} & d2 = {cur_date}.')
        X_test, y_test = self.prepare_samples_single(
                cur_date, previous_dates_list[1]
        )
        
        return X_test, y_test
        



    def train_and_report(
            self,
            X_test, y_test
    ):
        '''
        Use classification models.
        '''

        pd = {'max_depth': 20, 'n_estimators': 200}

        rf = RandomForestClassifier(**pd)

        print('\n~~~\n > Training model...')

        print(f' ... on {len(self.y_train)} data.')

        rf.fit(self.X_train, self.y_train)

        print('\n~~~\n > Evaluating model...')

        predictions = rf.predict(X_test)

        print(classification_report(y_test, predictions))

        return rf



    def save_model(
            self, 
            model, date
    ):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        model_path = Path(self.output_dir) / f'mod_{date}.joblib'

        joblib.dump(model, model_path)

        print(f"\n+++\n < Model Saved @ {model_path}.")



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

        self.X_train, self.y_train = None, None

        for i, cur_date in enumerate(date_list):

            print('-' * 70)

            #If true, it means mrap is necessary
            print(f"\n{i+1}. Getting ready for {cur_date} ...")
            
            previous_dates_list = get_dates_within(
                datetime_list=date_list[i-2 : i], 
                current_datetime=cur_date, 
                N_days=self.max_lookback_days
            )

            if len(previous_dates_list) < 2:
                '''
                We need at least 2 days, 
                    >= 2 for training
                    the last date and cur for testing.
                '''

                print("Skipped.., no enough dates.")

                continue

            X_test, y_test = self.prepare_samples(
                cur_date,
                previous_dates_list
            )

            model = self.train_and_report(X_test, y_test)

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



        

        








        
        
        
        
    

    

