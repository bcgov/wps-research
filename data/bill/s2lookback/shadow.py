'''
A look back method.

Shadow masking.
'''

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import numpy as np

from s2lookback.base import LookBack
from s2lookback.utils import get_dates_within
from misc import writeENVI
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


@dataclass
class Shadow(LookBack):

    eps: float = 1e-6
    n_workers: int = 4  # tune to your CPU count

    def mask_engine(self, cur_date, previous_dates_list):
        if len(previous_dates_list) == 0:
            print(' >< Skipped ... No previous date to mrap.')
            return None

        image_dat_main = self.read_image(cur_date)
        main_htrim = image_dat_main
        shadow_mask_all = None

        for prev_date in previous_dates_list:
            image_dat_prev = self.read_image(prev_date)
            prev_htrim = image_dat_prev

            brightness1 = np.mean(main_htrim, axis=-1)
            brightness2 = np.mean(prev_htrim, axis=-1)
            dark = brightness1 < 0.6 * brightness2

            num = np.sum(main_htrim * prev_htrim, axis=-1)
            den = np.linalg.norm(main_htrim, axis=-1) * np.linalg.norm(prev_htrim, axis=-1) + self.eps
            cos_sim = num / den
            shape_similar = cos_sim > 0.95

            shadow_mask_cur = dark & shape_similar

            if shadow_mask_all is None:
                shadow_mask_all = shadow_mask_cur
            else:
                shadow_mask_all = np.logical_and(shadow_mask_all, shadow_mask_cur)

        return shadow_mask_all

    def save_shadow_png(self, shadow_mask, date, filepath):
        fig = Figure(figsize=(12, 10))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.imshow(shadow_mask, cmap='gray')
        ax.set_title(f"Shadow mask: {date}")
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(filepath)

    def _process_date(self, args):
        """Worker function for a single date — must be picklable."""
        i, cur_date, date_list = args

        previous_dates_list = get_dates_within(
            datetime_list=date_list[:i],
            current_datetime=cur_date,
            N_days=self.max_lookback_days
        )

        print(f" > Masking Shadows: {cur_date} ... on {len(previous_dates_list)} dates.")

        shadow_mask = self.mask_engine(cur_date, previous_dates_list)

        if shadow_mask is None:
            return

        image_path = Path(self.get_file_path(cur_date, 'image_path'))

        self.save_shadow_png(
            shadow_mask,
            cur_date,
            f'{self.output_dir}/{image_path.stem}.png'
        )

        writeENVI(
            output_filename=f'{self.output_dir}/{image_path.stem}.bin',
            data=shadow_mask,
            ref_filename=image_path,
            mode='new'
        )

    def mask(self):
        self.get_file_dictionary()
        date_list = list(self.file_dict.keys())

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

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
    mrap = Shadow(
        image_dir='C11659/L1C/resampled_20m',
        output_dir='C11659/wps_shadow',
        max_lookback_days=8,
        n_workers=12
    )
    mrap.mask()