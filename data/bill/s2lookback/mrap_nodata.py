
'''
A look back method.

Masked to No-Data: sets all masked pixels (cloud/shadow) and existing
no-data (all-zero) pixels to zero, then saves.
'''

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from s2lookback.base import LookBack
from misc import writeENVI
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


@dataclass
class MASK_TO_NODATA(LookBack):

    def __post_init__(self):

        os.makedirs(self.output_dir, exist_ok=True)

        self.file_dict = self.get_file_dictionary()


    def save_png(self, raw, cleaned, date, filepath):

        fig = Figure(figsize=(19, 10))
        FigureCanvas(fig)
        axes = fig.subplots(1, 2)

        if self.lo is None and self.hi is None:
            lo = np.nanpercentile(raw, 1, axis=(0, 1))
            hi = np.nanpercentile(raw, 99, axis=(0, 1))
        else:
            lo, hi = self.lo, self.hi

        axes[0].imshow(np.clip((raw - lo) / (hi - lo), 0, 1))
        axes[0].set_title(f"Raw: {date}")
        axes[0].axis("off")

        axes[1].imshow(np.clip((cleaned - lo) / (hi - lo), 0, 1))
        axes[1].set_title(f"Cleaned: {date}")
        axes[1].axis("off")

        fig.tight_layout()
        fig.savefig(filepath)



    def _process_date(self, args):

        i, cur_date = args

        mask_dat, coverage = self.read_mask(cur_date)
        image_dat = self.read_image(cur_date)

        print(f"{i+1}. Processing {cur_date} (mask coverage: {coverage:.3f}) ...")

        cleaned = image_dat.copy()

        # Zero out masked pixels (cloud/shadow) and existing no-data
        nodata_mask = self.get_nodata_mask(image_dat)
        cleaned[mask_dat | nodata_mask] = 0

        image_path = Path(self.get_file_path(cur_date, 'image_path'))

        writeENVI(
            output_filename=f'{self.output_dir}/{image_path.stem}_cloudfree.bin',
            data=cleaned,
            ref_filename=image_path,
            mode='new',
            same_hdr=True
        )

        self.save_png(
            image_dat[..., :3],
            cleaned[..., :3],
            cur_date,
            f'{self.output_dir}/{cur_date}_CLOUDLESS.png'
        )

        print(f" < Done {cur_date}")



    def run(self):

        date_list = list(self.file_dict.keys())

        tasks = [(i, date) for i, date in enumerate(date_list)]

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(self._process_date, task): task[1] for task in tasks}
            for future in as_completed(futures):
                cur_date = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f" !! Error on {cur_date}: {e}")


if __name__ == "__main__":

    cleaner = MASK_TO_NODATA(
        image_dir='C11659/L1C/resampled_20m',
        mask_dir=['C11659/wps_cloud/merged_cloud', 'C11659/wps_shadow'],
        output_dir='C11659/wps_cloud/nodata_mrap',
        n_workers=12,
    )

    cleaner.run()