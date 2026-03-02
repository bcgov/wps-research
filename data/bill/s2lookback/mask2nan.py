
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


    def save_png(
            self, 
            raw, cleaned, 
            image_path):

        fig = Figure(figsize=(38, 20))
        FigureCanvas(fig)
        axes = fig.subplots(1, 2)

        if self.lo is None and self.hi is None:
            lo = np.nanpercentile(raw, 1, axis=(0, 1))
            hi = np.nanpercentile(raw, 99, axis=(0, 1))
        else:
            lo, hi = self.lo, self.hi

        axes[0].imshow(np.clip((raw - lo) / (hi - lo), 0, 1))
        axes[0].axis("off")

        axes[1].imshow(np.clip((cleaned - lo) / (hi - lo), 0, 1))
        axes[1].axis("off")

        big_title = "_".join([image_path.split("_")[2], image_path.split("_")[6], image_path])

        fig.suptitle(f'{image_path}_cloudfree.bin')

        fig.tight_layout()

        fig.savefig(f'{self.output_dir}/{big_title}_cloudfree.png')



    def _process_date(self, args):

        i, cur_date = args

        mask_dat, _ = self.read_mask(cur_date)
        image_dat = self.read_image(cur_date)

        print(f"[Processing] {cur_date}.")

        cleaned = image_dat.copy()

        cleaned[mask_dat] = np.nan

        image_path = Path(self.get_file_path(cur_date, 'image_path'))

        writeENVI(
            output_filename=f'{self.output_dir}/{image_path.stem}_cloudfree.bin',
            data=cleaned,
            ref_filename=image_path,
            mode='new',
            same_hdr=True,
            nodata_val=np.nan
        )

        self.save_png(
            image_dat[..., :3],
            np.nan_to_num(cleaned[..., :3], nan=0.0),
            image_path.stem
        )

        print(f"[DONE] {cur_date}")



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
        image_dir='C11659/L1C',
        mask_dir=['C11659/wps_inference/abcd_cloud_100', 'C11659/shadow', 'C11659/nodata'],
        output_dir='C11659/wps_inference/mask_removed',
        n_workers=12,
        mask_threshold=0.0005
    )

    cleaner.run()