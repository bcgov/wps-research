'''
A look back method.

Most recent available pixels (MRAP)
'''


from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import resource
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from s2lookback.base import LookBack
from s2lookback.utils import get_dates_within
from misc import writeENVI
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.linear_model import LinearRegression


@dataclass
class MRAP(LookBack):

    adjust_lighting: bool = False

    max_cov_for_adjust_lighting: float = 0.7

    min_adjust_sample_size: int = 5_000

    def __post_init__(self):

        os.makedirs(self.output_dir, exist_ok=True)

        self.file_dict = self.get_file_dictionary()



    def fill_engine(
        self, 
        image_dat_main: np.ndarray, 
        mask_dat_main: np.ndarray,
        previous_dates_list: list[datetime],
        adjust_lighting: bool, 
        sample_frac: float = 0.9
    ):

        if len(previous_dates_list) > 0:

            n_bands = image_dat_main.shape[-1]

            for prev_date in previous_dates_list[::-1]:

                # nodata_main = self.get_nodata_mask(image_dat_main, nodata_val=0.0)
            
                image_dat_prev = self.read_image(prev_date)

                # nodata_prev = self.get_nodata_mask(image_dat_prev, nodata_val=0.0)

                mask_dat_prev, _ = self.read_mask(prev_date)

                #This mask tells where to fill in past pixels, adjust if necessary.
                fill_mask = np.logical_and(
                    mask_dat_main,
                    ~mask_dat_prev
                )

                if ( adjust_lighting ):
 
                    # Overlap mask: where it is not masked (and no data) in main and in prev.
                    overlap_mask = np.logical_and(
                        ~mask_dat_main,
                        ~mask_dat_prev
                    )

                    overlap_indices = np.where(overlap_mask.ravel())[0]

                    if len(overlap_indices) > self.min_adjust_sample_size:

                        '''We sample just a fraction of it to learn the adjustment'''

                        n_samples = min(
                            self.min_adjust_sample_size, 
                            int(len(overlap_indices) * sample_frac)
                        )

                        sampled_indices = np.random.choice(overlap_indices, size=n_samples, replace=False)

                        # Sample pixels from both images: shape (n_samples, n_bands)
                        main_flat  = image_dat_main.reshape(-1, n_bands)
                        prev_flat  = image_dat_prev.reshape(-1, n_bands)

                        X = prev_flat[sampled_indices]   # predictor: prev image
                        y = main_flat[sampled_indices]   # target:    main image

                        # Fit one LinearRegression per band, then apply to fill pixels
                        fill_indices = np.where(fill_mask.ravel())[0]

                        if len(fill_indices) == 0:
                            continue  # nothing to fill for this date, move to next

                        adjusted_fill = np.empty((len(fill_indices), n_bands), dtype=image_dat_main.dtype)

                        for b in range(image_dat_main.shape[-1]):

                            reg = LinearRegression()
                            reg.fit(X.reshape(-1, n_bands), y[:, b])

                            adjusted_fill[:, b] = reg.predict(
                                prev_flat[fill_indices]
                            ).\
                                astype(image_dat_main.dtype)

                        image_dat_main.reshape(-1, n_bands)[fill_indices] = adjusted_fill

                    else:
                        # Not enough overlap to fit — fall back to direct fill
                        image_dat_main[fill_mask] = image_dat_prev[fill_mask]

                else:
                    image_dat_main[fill_mask] = image_dat_prev[fill_mask]


                mask_dat_main = np.logical_and(mask_dat_main, ~fill_mask)

        # After all filling: any pixel still masked, force to nodata
        image_dat_main[mask_dat_main] = np.nan

        return image_dat_main



    def _save_png(
            self, 
            mrap: np.ndarray, 
            date: datetime, 
            filepath: str
    ):

        fig = Figure(figsize=(10, 10))
        FigureCanvas(fig)
        ax = fig.subplots(1, 1)

        if not (hasattr(self, 'lo') and hasattr(self, 'hi')):
            lo = np.nanpercentile(mrap, 1, axis=(0, 1))
            hi = np.nanpercentile(mrap, 99, axis=(0, 1))
        else:
            lo, hi = self.lo, self.hi

        ax.imshow(np.clip((mrap - lo) / (hi - lo), 0, 1))
        ax.set_title(f"Composite: {date}")
        ax.axis("off")

        fig.tight_layout()
        fig.savefig(filepath)



    def _process_date(self, args):
        
        """Worker for a single date. Each date writes its own output — no shared state."""
        i, cur_date, date_list = args

        mask_dat_main, coverage = self.read_mask(cur_date)
        image_dat_main = self.read_image(cur_date)

        adjust_lighting = self.adjust_lighting & (coverage <= self.max_cov_for_adjust_lighting)

        print(f"\n[PROCESSING] {cur_date} | Adj. lighting: {adjust_lighting}")

        previous_dates_list = get_dates_within(
            datetime_list=date_list[:i],
            current_datetime=cur_date,
            N_days=self.max_lookback_days
        )

        mrap_dat = self.fill_engine(
            image_dat_main.copy(),
            mask_dat_main,
            previous_dates_list,
            adjust_lighting = adjust_lighting
        )

        image_path = Path(self.get_file_path(cur_date, 'image_path'))

        writeENVI(
            output_filename=f'{self.output_dir}/{image_path.stem}_cloudfree.bin',
            data=mrap_dat,
            ref_filename=image_path,
            mode='new',
            same_hdr=True,
            nodata_val=np.nan
        )

        self._save_png(
            np.nan_to_num(mrap_dat[..., :3], nan=0.0),
            cur_date,
            f'{self.output_dir}/{cur_date}_cloudfree.png'
        )

        print(f"\n[DONE] {cur_date}.")



    def fill(self):

        date_list = list(self.file_dict.keys())

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        # Raise file descriptor limit before spawning processes
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

        tasks = [(i, cur_date, date_list) for i, cur_date in enumerate(date_list)]

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(self._process_date, task): task[1] for task in tasks}
            for future in as_completed(futures):
                cur_date = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"\n! Error on {cur_date}: {e}")



if __name__ == "__main__":

    mrap = MRAP(
        image_dir='C11659/L1C',
        mask_dir=['C11659/wps_inference/abcd_cloud_100', 'C11659/shadow', 'C11659/nodata'],
        output_dir='C11659/wps_inference/mrap_with_correction',
        max_lookback_days=60,
        n_workers=8,
        adjust_lighting=True,
        mask_threshold=0.0001
    )

    mrap.fill()
