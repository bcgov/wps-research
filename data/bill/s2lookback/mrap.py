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

    def __post_init__(self):

        os.makedirs(self.output_dir, exist_ok=True)

        self.file_dict = self.get_file_dictionary()



    def fill_engine(
        self, 
        image_dat_main, 
        mask_dat_main,
        previous_dates_list,
        sample_frac: float = 0.1
    ):
        
        if len(previous_dates_list) == 0:

            print(' >< Skipped ... No previous date to mrap.')

        else:

            for prev_date in previous_dates_list[::-1]:

                print('filling', prev_date)

                nodata_main = self.get_nodata_mask(image_dat_main, nodata_val=0.0)
            
                image_dat_prev = self.read_image(prev_date)

                nodata_prev = self.get_nodata_mask(image_dat_prev, nodata_val=0.0)

                mask_dat_prev, _ = self.read_mask(prev_date)

                fill_mask = np.logical_and(
                    np.logical_or(mask_dat_main, nodata_main),
                    ~np.logical_or(mask_dat_prev, nodata_prev)
                )

                if self.adjust_lighting:
                    # Overlap mask: valid (unmasked, non-nodata) in both images
                    overlap_mask = (
                        ~np.logical_or(mask_dat_main, nodata_main) &
                        ~np.logical_or(mask_dat_prev, nodata_prev)
                    )

                    overlap_indices = np.where(overlap_mask.ravel())[0]

                    if len(overlap_indices) > 1:
                        n_samples = max(1, int(len(overlap_indices) * sample_frac))
                        sampled_indices = np.random.choice(overlap_indices, size=n_samples, replace=False)

                        # Sample pixels from both images: shape (n_samples, n_bands)
                        main_flat  = image_dat_main.reshape(-1, image_dat_main.shape[-1])
                        prev_flat  = image_dat_prev.reshape(-1, image_dat_prev.shape[-1])

                        X = prev_flat[sampled_indices]   # predictor: prev image
                        y = main_flat[sampled_indices]   # target:    main image

                        # Fit one LinearRegression per band, then apply to fill pixels
                        fill_indices = np.where(fill_mask.ravel())[0]
                        adjusted_fill = np.empty((len(fill_indices), image_dat_main.shape[-1]), dtype=image_dat_main.dtype)

                        for b in range(image_dat_main.shape[-1]):
                            reg = LinearRegression()
                            reg.fit(X[:, b].reshape(-1, 1), y[:, b])
                            adjusted_fill[:, b] = reg.predict(
                                prev_flat[fill_indices, b].reshape(-1, 1)
                            ).astype(image_dat_main.dtype)

                        image_dat_main.reshape(-1, image_dat_main.shape[-1])[fill_indices] = adjusted_fill

                    else:
                        # Not enough overlap to fit — fall back to direct fill
                        image_dat_main[fill_mask] = image_dat_prev[fill_mask]

                else:
                    image_dat_main[fill_mask] = image_dat_prev[fill_mask]

                mask_dat_main = np.logical_and(mask_dat_main, ~fill_mask)

        # After all filling: any pixel still masked or all-zero → force to nodata
        current_coverage = mask_dat_main.mean()
        remaining_nodata = np.logical_or(mask_dat_main, self.get_nodata_mask(image_dat_main, nodata_val=0.0))
        image_dat_main[remaining_nodata] = 0   # ensure truly zeroed

        return image_dat_main, current_coverage



    def save_old_new(self, mrap, date, filepath):

        fig = Figure(figsize=(10, 10))
        FigureCanvas(fig)
        ax = fig.subplots(1, 1)

        if self.lo is None and self.hi is None:
            lo = np.nanpercentile(mrap, 1, axis=(0, 1))
            hi = np.nanpercentile(mrap, 99, axis=(0, 1))
        else:
            lo, hi = self.lo, self.hi

        ax.imshow(np.clip((mrap - lo) / (hi - lo), 0, 1))
        ax.set_title(f"Final Composite: {date}")
        ax.axis("off")

        fig.tight_layout()
        fig.savefig(filepath)



    def _process_date(self, args):
        
        """Worker for a single date. Each date writes its own output — no shared state."""
        i, cur_date, date_list = args

        mask_dat_main, coverage = self.read_mask(cur_date)
        image_dat_main = self.read_image(cur_date)

        print('-' * 50)

        if coverage >= self.max_mask_prop_usable:
            print(f"{i+1}. >< Skipped, current mask coverage {coverage:.3f} is more than usable of maximum {self.min_mask_prop_trigger} -> Useless.")
            return

        print(f" > Filling {cur_date} ...")

        previous_dates_list = get_dates_within(
            datetime_list=date_list[:i],
            current_datetime=cur_date,
            N_days=self.max_lookback_days
        )

        mrap_dat, post_fill_coverage = self.fill_engine(
            image_dat_main.copy(),
            mask_dat_main,
            previous_dates_list
        )

        image_path = Path(self.get_file_path(cur_date, 'image_path'))

        writeENVI(
            output_filename=f'{self.output_dir}/{image_path.stem}_MRAP.bin',
            data=mrap_dat,
            ref_filename=image_path,
            mode='new',
            same_hdr=True
        )

        self.save_old_new(
            # image_dat_main[..., :3],
            mrap_dat[..., :3],
            cur_date,
            f'{self.output_dir}/{cur_date}_MRAP.png'
        )

        print(f" < Completed {cur_date}, coverage changed from {coverage:.3f} to {post_fill_coverage:.3f}")



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
                    print(f" !! Error on {cur_date}: {e}")



if __name__ == "__main__":

    mrap = MRAP(
        image_dir='C11659/L1C/resampled_20m',
        mask_dir='C11659/wps_inference/abcd_cloud',
        output_dir='C11659/wps_inference/abcd_mrap',
        max_lookback_days=45,
        n_workers=12,
        adjust_lighting=True
    )

    mrap.fill()
