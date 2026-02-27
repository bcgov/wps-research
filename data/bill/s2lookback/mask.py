'''
A look back method.

This module attempts to mask cloud, cloud shadow.
'''

from s2lookback.base import LookBack

import os

from tqdm import tqdm

import joblib

from pathlib import Path

import numpy as np

from datetime import datetime

from dataclasses import dataclass

from plot_tools import plot_multiple 

from misc import writeENVI

#Machine Learning**
from cuml.ensemble import RandomForestClassifier
from cuml.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

@dataclass
class MASK(LookBack):

    model_path: str = None
    progressive_testing: bool = True
    prediction_threshold: float = 0.5
    merge_mask: bool = False
    sample_size: int = 5_000
    min_lighting_samples = 5_000
    n_feature: int = 3

    # Lighting normalization — single slot, overwritten per date
    lighting_ref_date: datetime = None          # set automatically in __post_init__


    def __post_init__(self):

        self.validate_method()

        Path(self.output_dir).mkdir(exist_ok=True)

        self.file_dict = self.get_file_dictionary()

        if self.model_path is not None:
            self.model = joblib.load(self.model_path)

        # Single-slot lighting state (not dataclass fields — set imperatively)
        self._lighting_model_date  = None
        self._lighting_model_bands = None

        self._find_lighting_reference()



    def validate_method(self):

        if self.sample_size is None:
            raise ValueError('This method requires a sample size, received None.')



    def _find_lighting_reference(self):

        """
        Find the clearest date and cache its image for repeated fitting.
        """

        if (self.lighting_ref_date is None):

            best_date, best_coverage = None, float('inf')

            for date in tqdm(self.file_dict.keys(), desc='[Lighting] Scanning mask coverage'):
                _, coverage = self.read_mask(date)
                if coverage < best_coverage:
                    best_coverage = coverage
                    best_date = date

            self.lighting_ref_date = best_date

        else:

            _, best_coverage = self.read_mask(self.lighting_ref_date)

        print(f"\n[Lighting] Reference date: {self.lighting_ref_date} (mask coverage: {best_coverage:.3f})")

        ref_img_raw     = self.read_image(self.lighting_ref_date)
        self._ref_img   = ref_img_raw.astype(np.float32) / 10_000.
        ref_mask, _     = self.read_mask(self.lighting_ref_date)
        ref_nodata      = self.get_nodata_mask(ref_img_raw)
        self._ref_valid = ~np.logical_or(ref_mask, ref_nodata)



    def _fit_lighting_model(
        self, 
        date: datetime, 
        sample_frac: float = 0.1
    ):
        
        """
        Fit and overwrite the single per-band lighting model for `date`.
        Maps this date's pixels → reference date's pixels on clean overlap regions.
        """

        if date == self.lighting_ref_date:
            self._lighting_model_date  = date
            self._lighting_model_bands = None   # identity — no transform needed
            return

        img_raw = self.read_image(date)
        img     = img_raw.astype(np.float32) / 10_000.
        mask, _ = self.read_mask(date)
        nodata  = self.get_nodata_mask(img_raw)
        valid   = ~np.logical_or(mask, nodata)

        overlap     = self._ref_valid & valid
        overlap_idx = np.where(overlap.ravel())[0]

        img_flat = img.reshape(-1, img.shape[-1])
        ref_flat = self._ref_img.reshape(-1, self._ref_img.shape[-1])

        if len(overlap_idx) < self.min_lighting_samples:
            print(f"[Lighting] {date}: insufficient overlap ({len(overlap_idx)} px), no normalizer fitted.")
            self._lighting_model_date  = date
            self._lighting_model_bands = None
            return

        n_samples = min(self.min_lighting_samples, int(len(overlap_idx) * sample_frac))
        sampled   = np.random.choice(overlap_idx, size=n_samples, replace=False)

        band_models = []
        for b in range(img.shape[-1]):
            reg = LinearRegression()
            reg.fit(img_flat[sampled, b].reshape(-1, 1), ref_flat[sampled, b])
            band_models.append(reg)

        self._lighting_model_date  = date
        self._lighting_model_bands = band_models

        print(f"[Lighting] {date}: model fitted on {n_samples} overlap samples.")



    def data_engineering(
            self, 
            data: np.ndarray, 
            date: datetime = None
    ):

        img = data.astype(np.float32) / 10_000.

        if (
            date is not None
            and date == self._lighting_model_date
            and self._lighting_model_bands is not None
        ):
            h, w, nb = img.shape
            flat = img.reshape(-1, nb).copy()
            for b, reg in enumerate(self._lighting_model_bands):
                flat[:, b] = reg.predict(flat[:, b].reshape(-1, 1)).ravel()
            img = flat.reshape(h, w, nb)

        return img


    def prepare_samples(
            self, 
            date: datetime
    ):

        img_dat     = self.read_image(date)
        nodata_mask = self.get_nodata_mask(img_dat)
        mask, mask_coverage = self.read_mask(date)

        if mask_coverage < self.min_mask_prop_trigger:
            print(f'Skipped.., coverage of {mask_coverage} < {self.min_mask_prop_trigger}.')
            raise ValueError('Coverage does not satisfy.')

        #We might really need this, because the lighting condition changes everyday.
        self._fit_lighting_model(date)

        IMG_DAT = self.data_engineering(img_dat, date=date)

        VALID_IMG_DAT = IMG_DAT[~nodata_mask]
        VALID_MASK_CUR = mask[~nodata_mask]

        if self.progressive_testing:
            self.cur_date    = date
            self.cur_img_dat = img_dat   # store raw so plot_classification can re-apply normalisation

        return self.sample_datasets(VALID_IMG_DAT, VALID_MASK_CUR)
    


    def prepare_train_test(self, date: datetime, test_data=False):

        print(f'\n***\n > Sampling Data: {date}')

        X_train, y_train = self.prepare_samples(date)

        X_test, y_test = None, None

        if self.progressive_testing or test_data:
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, shuffle=True)

        if self.X_train is None:
            self.X_train = X_train
            self.y_train = y_train
        else:
            self.X_train = np.vstack([self.X_train, X_train])
            self.y_train = np.concatenate([self.y_train, y_train])

        return X_test, y_test
    


    def train_and_report(
            self, 
            X_test: np.ndarray, 
            y_test: np.ndarray
    ):

        print(f'\n~~~\n > Fitting classifier on {len(self.y_train)} data.')

        clf = RandomForestClassifier(max_depth=20, n_estimators=200)
        # clf = LogisticRegression(max_iter=1000)

        clf.fit(self.X_train, self.y_train)

        print('\n~~~\n > Evaluating model...')

        predictions = clf.predict_proba(X_test)

        print(classification_report(y_test, predictions[:, 1] > self.prediction_threshold))

        self.model = clf



    def save_model(
            self, 
            date: datetime
    ):

        model_dir = Path(self.output_dir) / 'models'

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        model_path = model_dir / f'{date}.joblib'
        joblib.dump(self.model, model_path)

        print(f"\n+++\n < Model Saved @ {model_path}.")



    def plot_classification(self):

        sen2cor_mask, _ = self.read_mask(self.cur_date, as_prob=True)

        # cur_date's lighting model is still loaded in the single slot at this point
        X = self.data_engineering(self.cur_img_dat, date=self.cur_date)

        predicted_map = self.model.predict_proba(X.reshape(-1, self.n_feature))

        plot_multiple(
            [sen2cor_mask, predicted_map[:, 1].reshape(sen2cor_mask.shape)],
            title_list=['Sen2Cor Mask', 's2lookback (BC Wildfire Service)'],
            suptitle=f'Tested Masking on {self.cur_date}',
            figsize=(18, 10)
        )



    def mask_and_save(self):

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        date_list = list(self.file_dict.keys())

        for date in date_list:

            print(" > Masking:", date)

            img_dat = self.read_image(date)
            mask, _ = self.read_mask(date)

            # Fit lighting model for this date before inference
            self._fit_lighting_model(date)

            X = self.data_engineering(img_dat, date=date).reshape(-1, self.n_feature)

            predictions = self.model.predict_proba(X)[:, 1].reshape(mask.shape)

            fig = Figure(figsize=(19, 10))
            canvas = FigureCanvas(fig)
            axes = fig.subplots(1, 2)

            axes[0].imshow(mask, cmap="gray")
            axes[0].set_title("Sen2Cor (ESA)")
            axes[0].axis("off")

            axes[1].imshow(predictions, cmap="gray")
            axes[1].set_title("s2lookback (BC Wildfire Service)")
            axes[1].axis("off")

            fig.tight_layout()

            file_path = Path(self.get_file_path(date, 'image_path'))

            fig.savefig(Path(self.output_dir) / f"{file_path.stem}.png")

            writeENVI(
                output_filename=Path(self.output_dir) / f"{file_path.stem}.bin",
                data=predictions,
                ref_filename=file_path,
                mode='new'
            )



    def fit(
            self
    ):

        date_list = list(self.file_dict.keys())

        self.X_train, self.y_train = None, None

        for i, date in enumerate(date_list):

            print('-' * 100)

            #If true, it means mrap is necessary
            print(f"\n{i+1}. Processing {date} ...")
            
            try:
                X_test, y_test = self.prepare_train_test(date, i == len(date_list) - 1)

            except Exception:

                continue

            if (
                self.progressive_testing or \
                i == len(date_list) - 1
            ):
                
                #Model fitting.
                self.train_and_report(X_test, y_test)

                #Save the model.
                self.save_model(date)

                #Temp test.
                self.plot_classification()


    
    def transform(
            self
    ):
        
        self.mask_and_save()
        


if __name__ == "__main__":

    masker = MASK(
        image_dir='C11659/L1C/resampled_20m',
        mask_dir='C11659/cloud_20m',
        model_path='C11659/wps_inference/cloud_2/models/2025-09-09 19:19:09.joblib',
        output_dir='C11659/wps_inference/cloud_2',
        progressive_testing=False,
        lighting_ref_date=datetime(2025,8,23,19,29,9),
        start=datetime(2025,8,20),
        end=datetime(2025,9,10)
    )

    # masker.fit()

    masker.transform()



        

        








        
        
        
        
    

    

