'''
A look back method.

This module attempts to mask cloud, cloud shadow.
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

from plot_tools import plot_multiple 

from misc import writeENVI


@dataclass
class MASK(LookBack):

    only_test_last_date: bool = False #Only fit the model on the last date.

    mask_all_final: bool = False

    n_feature: int = 12


    def __post_init__(self):
        
        self.validate_method()

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.file_dict = self.get_file_dictionary()



    def validate_method(
            self
    ):
        '''
        Checks for all conditions before running.
        '''
        if self.sample_size is None:

            raise ValueError('This method requires a sample size, received None.')
    


    def data_engineering(
            self,
            dat_cur: np.ndarray,
            dat_prev: np.ndarray
    ):
        
        DIFF = (dat_cur - dat_prev) / (dat_cur + dat_prev + 1e-3)

        IMG_DAT = np.dstack([DIFF, dat_cur / 10_000, dat_prev / 10_000])

        return IMG_DAT
        


    def prepare_samples(
            self, 
            cur_date: datetime,
            prev_date: datetime,
            for_test: bool = False
    ):
        '''
        Helper for prepare samples
        '''

        img_dat_cur = self.read_image(cur_date)
        nodata_cur = np.all(img_dat_cur == 0, axis=-1)

        img_dat_prev = self.read_image(prev_date)
        nodata_prev = np.all(img_dat_prev == 0, axis=-1)

        mask_cur, mask_coverage = self.read_mask(cur_date)

        if mask_coverage < self.min_mask_prop_trigger:
            print(f'Skipped.., coverage of {mask_coverage} < {self.min_mask_prop_trigger}.')
            raise ValueError('Coverage does not satisfy.') 
        
        ALL_NODATA_MASK = np.logical_or(nodata_prev, nodata_cur)

        #Data Engineering (calling function)

        IMG_DAT = self.data_engineering(img_dat_cur, img_dat_prev)

        #Filters no data part, useless for training.

        VALID_IMG_DAT = IMG_DAT[~ALL_NODATA_MASK]

        VALID_MASK_CUR = mask_cur[~ALL_NODATA_MASK]

        #Temporary, use this and directly plot
        if not hasattr(self, "n_feature"):

            self.n_feature = VALID_IMG_DAT.shape[1]

        if for_test:

            self.cur_img = IMG_DAT.reshape(-1, self.n_feature)
            self.cur_mask = mask_cur
            self.cur_date = cur_date

        return self.sample_datasets(VALID_IMG_DAT, VALID_MASK_CUR)
    


    def prepare_train_test(
            self,
            cur_date: datetime,
            previous_dates_list: list[datetime],
            get_test_data: bool
    ):

        #Sample for train data
        prev_, cur_ = previous_dates_list[0], previous_dates_list[1]

        print(f'\n***\n > Sampling TRAIN data: d1 = {prev_} & d2 = {cur_}.')

        X_train, y_train = self.prepare_samples(
            cur_, prev_
        )

        if self.X_train is None: 
            self.X_train = X_train
            self.y_train = y_train

        else:
            self.X_train = np.vstack([self.X_train, X_train])
            self.y_train = np.concatenate([self.y_train, y_train])

        #Sample for test data
        X_test, y_test = None, None

        if get_test_data:

            print(f'\n***\n > Sampling TEST data: d1 = {previous_dates_list[1]} & d2 = {cur_date}.')
            X_test, y_test = self.prepare_samples(
                    cur_date, previous_dates_list[1], 
                    for_test=True
            )
            
        return X_test, y_test
            


    def train_and_report(
            self,
            X_test, y_test
    ):
        '''
        Trains and returns classification models.

        Not restricted to Random Forests, can try other classification models.
        '''

        pd = {'max_depth': 20, 'n_estimators': 200}

        clf = RandomForestClassifier(**pd)

        print(f'\n~~~\n > Fitting classifier on {len(self.y_train)} data.')

        clf.fit(self.X_train, self.y_train)

        print('\n~~~\n > Evaluating model...')

        predictions = clf.predict_proba(X_test)

        print(classification_report(y_test, predictions[:, 1] > .5))

        return clf
    


    def plot_classification(
            self,
            model
    ):
        '''
        Use current model to predict and plot.
        '''

        predicted_map = model.predict_proba(self.cur_img)

        sen2cor_map = self.cur_mask

        plot_multiple(
            [sen2cor_map, predicted_map[:, 1].reshape(sen2cor_map.shape)],
            title_list=['Sen2Cor (ESA)', 's2lookback (BC Wildfire Service)'],
            suptitle=f'Tested Masking on {self.cur_date}',
            figsize=(18,10)
        )



    def save_model(
            self, 
            model, date: datetime
    ):
        
        model_dir = Path(self.output_dir) / 'models'
  
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        model_path = model_dir / f'{date}.joblib'

        joblib.dump(model, model_path)

        print(f"\n+++\n < Model Saved @ {model_path}.")



    def mask_all_and_save_png(
            self,
            date_list: list[datetime],
            model
    ):
        '''
        Masks all data and saves.

        Using threads.
        '''

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        predicted_mask_dir = Path(self.output_dir) / "predictions"
        predicted_mask_dir.mkdir(exist_ok=True)

        for i in range(len(date_list) - 1):

            prev_date = date_list[i]
            cur_date = date_list[i + 1]

            print("Processing:", cur_date)

            # ---- Load ----
            img_dat_cur = self.read_image(cur_date)
            img_dat_prev = self.read_image(prev_date)
            mask_cur, _ = self.read_mask(cur_date)

            # ---- Feature Engineering ----
            X = self.data_engineering(
                img_dat_cur, img_dat_prev
            ).reshape(-1, self.n_feature)

            predictions = model.predict_proba(X)[:, 1]
            predictions = predictions.reshape(mask_cur.shape)

            save_path = predicted_mask_dir / f"{cur_date}.png"

            fig = Figure(figsize=(19, 10))
            canvas = FigureCanvas(fig)
            axes = fig.subplots(1, 2)

            axes[0].imshow(mask_cur, cmap="gray")
            axes[0].set_title("Sen2Cor (ESA)")
            axes[0].axis("off")

            axes[1].imshow(predictions, cmap="gray")
            axes[1].set_title("s2lookback (BC Wildfire Service)")
            axes[1].axis("off")

            fig.tight_layout()
            fig.savefig(save_path)



    def mask_and_merge(
            self,
            date_list: list[datetime],
            model
    ):
        '''
        Masks all data and saves.

        Using threads.
        '''

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        predicted_mask_dir = Path(self.output_dir) / "merged_mask"
        predicted_mask_dir.mkdir(exist_ok=True)

        for i in range(len(date_list) - 1):

            prev_date = date_list[i]
            cur_date = date_list[i + 1]

            print("Processing:", cur_date)

            # ---- Load ----
            img_dat_cur = self.read_image(cur_date)
            img_dat_prev = self.read_image(prev_date)
            mask_cur, _ = self.read_mask(cur_date)

            # ---- Feature Engineering ----
            X = self.data_engineering(
                img_dat_cur, img_dat_prev
            ).reshape(-1, self.n_feature)

            predictions = model.predict_proba(X)[:, 1]
            predictions = predictions.reshape(mask_cur.shape)

            #Merge predictions
            merged_predictions = np.logical_or(mask_cur, predictions > .3)

            fig = Figure(figsize=(19, 10))
            canvas = FigureCanvas(fig)
            axes = fig.subplots(1, 2)

            axes[0].imshow(mask_cur, cmap="gray")
            axes[0].set_title("Sen2Cor (ESA)")
            axes[0].axis("off")

            axes[1].imshow(merged_predictions, cmap="gray")
            axes[1].set_title("s2lookback (BC Wildfire Service) and Sen2Cor")
            axes[1].axis("off")

            fig.tight_layout()

            '''
            Save data, combining S2 file with tails.
            '''
            file_path = Path(self.get_file_path(cur_date, 'image_path'))

            #Save fig
            fig.savefig(predicted_mask_dir / f"{file_path.stem}.png")

            #Save ENVI
            writeENVI(
                output_filename = predicted_mask_dir / f"{file_path.stem}.bin",
                data = merged_predictions,
                ref_filename = file_path,
                mode = 'new'
            )



    def fit(
            self
    ):
        '''
        1. Match files in dictionary
        '''

        date_list = list(self.file_dict.keys())

        '''
        2. Iterate from the 2nd date to the final date. 

            We do not mrap on first date, because there was no data prior to the 1st date (dates were sorted).
        '''
        from s2lookback.utils import get_dates_within

        self.X_train, self.y_train = None, None

        for i, cur_date in enumerate(date_list):

            print('-' * 100)

            #If true, it means mrap is necessary
            print(f"\n{i+1}. Getting ready for {cur_date} ...")
            
            previous_dates_list = get_dates_within(
                datetime_list=date_list[i-2 : i], 
                current_datetime=cur_date, 
                N_days=5
            )

            if len(previous_dates_list) < 2:
                '''
                We need at least 2 days, 
                    >= 2 for training
                    the last date and cur for testing.
                '''

                print("Skipped.., not enough dates.")

                continue
            

            try:
                X_test, y_test = self.prepare_train_test(
                    cur_date,
                    previous_dates_list,
                    get_test_data = (not self.only_test_last_date) or (i == len(date_list) - 1)
                )

            except Exception:

                continue


            if (X_test is not None and y_test is not None):
                
                #Model fitting.
                model = self.train_and_report(X_test, y_test)

                #Save the model.
                self.save_model(model, cur_date)

                #Temp test.
                self.plot_classification(model)
        

        if self.mask_all_final:

            print("\n > Performing final masking on all dates ...")
            self.mask_all_and_save_png(date_list, model)


    
    def mask(
            self,
            model_path: str
    ):
        
        '''
        1. Match files in dictionary
        '''

        date_list = list(self.file_dict.keys())

        '''
        2. Load Model
        '''
        model = joblib.load(model_path)

        print("\n > Performing final masking on all dates ...")
        
        self.mask_and_merge(date_list, model)
        


if __name__ == "__main__":

    masker = MASK(
        image_dir='C11659/L1C/resampled_20m',
        mask_dir='C11659/cloud_20m',
        output_dir='C11659/wps_cloud_2',
        sample_size=100_000,
        only_test_last_date = True,
        mask_all_final = True
    )

    masker.mask('C11659/wps_cloud/models/cloud_model.joblib')



        

        








        
        
        
        
    

    

