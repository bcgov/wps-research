#!/usr/bin/env python3
'''
abcd_mask.py — Per-date "A:B::C:D" cloud masking via Random Forest regression.

Delegates all RF logic to abcd_rf.py (abcd_rf_arrays).

For each date independently:
    A = image         (band values)       — training features
    B = cloud prob    (mask probability)  — training targets
    C = image again   (same as A)         — inference features
    D = predicted cloud probability map   — saved as ENVI .bin/.hdr + .png

Each date's model is optionally cached as a .pkl inside output_dir/models/.

Usage
-----
    python3 abcd_mask.py <image_dir> <mask_dir> <output_dir> <skip_f> [offset]

Arguments
---------
    image_dir  — directory of satellite images  (acquisition time from metadata)
    mask_dir   — directory of cloud probability masks  (matched by acquisition time)
    output_dir — directory where .bin / .hdr / .png outputs are written
    skip_f     — spatial sampling stride  (e.g. 10 = every 10th pixel)
    offset     — sampling offset, default 0
'''

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from osgeo import gdal
except ImportError:
    import gdal

from s2lookback.base import LookBack
from machine_learning.abcd_rf import abcd_rf_arrays       # <-- all RF logic lives here

gdal.AllRegister()
gdal.UseExceptions()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_envi_prob(out_path: str, prob_1d: np.ndarray, ref_path: str):
    '''
    Write a single-band float32 ENVI raster (.bin + .hdr).

    Parameters
    ----------
    out_path : destination .bin path
    prob_1d  : flat (n_pixels,) array — predicted cloud probability
    ref_path : reference image — spatial reference copied from here
    '''
    ref_ds = gdal.Open(str(ref_path), gdal.GA_ReadOnly)
    if ref_ds is None:
        sys.exit(f'[ERROR] Cannot open reference raster: {ref_path}')

    nr, nc = ref_ds.RasterYSize, ref_ds.RasterXSize

    driver = gdal.GetDriverByName('ENVI')
    out_ds = driver.Create(str(out_path), nc, nr, 1, gdal.GDT_Float32)
    if out_ds is None:
        sys.exit(f'[ERROR] Cannot create output: {out_path}')

    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())

    band = out_ds.GetRasterBand(1)
    band.WriteArray(prob_1d.reshape(nr, nc))
    band.SetNoDataValue(float('nan'))

    out_ds.FlushCache()
    out_ds = None
    ref_ds = None


def save_png(out_path: str, prob_1d: np.ndarray, ref_path: str,
             date, sen2cor_prob: np.ndarray):
    '''
    Save a side-by-side PNG: Sen2Cor mask vs predicted probability.

    Parameters
    ----------
    out_path     : destination .png path
    prob_1d      : flat (n_pixels,) predicted cloud probability
    ref_path     : reference image path — used to get image shape
    date         : acquisition datetime — used in the plot title
    sen2cor_prob : flat (n_pixels,) Sen2Cor cloud probability
    '''
    ref_ds     = gdal.Open(str(ref_path), gdal.GA_ReadOnly)
    nr, nc     = ref_ds.RasterYSize, ref_ds.RasterXSize
    ref_ds     = None

    fig, axes  = plt.subplots(1, 2, figsize=(18, 8))

    axes[0].imshow(sen2cor_prob.reshape(nr, nc), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Sen2Cor (ESA)')
    axes[0].axis('off')

    axes[1].imshow(prob_1d.reshape(nr, nc), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('abcd_mask prediction')
    axes[1].axis('off')

    fig.suptitle(str(date), fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def per_date_pkl_path(output_dir: str, date, skip_f: int, offset: int) -> str:
    '''Return .pkl path for a given date inside output_dir/models/.'''
    model_dir  = Path(output_dir) / 'models'
    model_dir.mkdir(exist_ok=True)
    safe_date  = str(date).replace(' ', '_').replace(':', '-')
    return str(model_dir / f'{safe_date}_{skip_f}_{offset}.pkl')


# ---------------------------------------------------------------------------
# ABCD_MASK
# ---------------------------------------------------------------------------

@dataclass
class ABCD_MASK(LookBack):
    '''
    Per-date RF regression cloud masker.

    For every date:
      1. Read image (A) and cloud probability mask (B).
      2. Call abcd_rf_arrays(A, B, A, ...) to train RF on stride-sampled pixels
         and predict cloud probability for every pixel.
      3. Save the result as ENVI (.bin/.hdr) + PNG.

    All RF logic is handled by abcd_rf.abcd_rf_arrays — this class only
    manages file I/O and the per-date loop.
    '''

    skip_f:      int  = 5_000
    offset:      int  = 0
    save_model:  bool = True


    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.file_dict = self.get_file_dictionary()


    def mask(self):
        '''
        Main loop — for every date: run abcd → save ENVI + PNG.
        '''
        dates = list(self.file_dict.keys())
        print(f'\n[ABCD_MASK] Processing {len(dates)} dates ...\n')

        for date in tqdm(dates, desc='[ABCD_MASK]'):

            print(f'\n--- {date} ---')

            img_path = self.get_file_path(date, 'image_path')
            ref_path = img_path if isinstance(img_path, str) else img_path[0]
            stem     = Path(ref_path).stem

            # ---- prepare A and B as (n_bands, n_pixels) float32 arrays ----

            img_raw       = self.read_image(date)                  # (H, W, B)
            nodata        = self.get_nodata_mask(img_raw)          # (H, W)
            mask_prob, _  = self.read_mask(date, as_prob=True)     # (H, W)

            # Scale to [0,1] and reshape to (n_bands, n_pixels)
            A = (img_raw.astype(np.float32) / 10_000.)
            A = A.reshape(-1, A.shape[-1]).T                       # (B, n_px)

            # Mark nodata as NaN so abcd_rf_arrays excludes them automatically
            nodata_flat = nodata.ravel()
            A[:, nodata_flat] = np.nan

            # B is cloud probability — shape (1, n_pixels)
            B = mask_prob.ravel().astype(np.float32)[np.newaxis, :]  # (1, n_px)
            np.nan_to_num(B, copy=False, nan=0.0)

            # C is the same image (A:B :: A:D)
            C = A

            # ---- pkl path (None = don't cache) ----
            pkl = (per_date_pkl_path(self.output_dir, date, self.skip_f, self.offset)
                   if self.save_model else None)

            # ---- delegate RF logic entirely to abcd_rf ----
            try:
                D, _ = abcd_rf_arrays(A, B, C,
                                      skip_f=self.skip_f,
                                      offset=self.offset,
                                      pkl_path=pkl)
            except ValueError as e:
                print(f'  [SKIP] {e}')
                continue

            # D shape is (1, n_pixels) — extract the single band
            prob = np.clip(D[0], 0.0, 1.0)

            # ---- save ENVI ----
            out_bin = Path(self.output_dir) / f'{stem}.bin'
            write_envi_prob(str(out_bin), prob, ref_path)
            print(f'  Saved ENVI → {out_bin}')

            # ---- save PNG ----
            out_png = Path(self.output_dir) / f'{stem}.png'
            save_png(str(out_png), prob, ref_path, date, B[0])
            print(f'  Saved PNG  → {out_png}')

        print(f'\n[Done] All outputs written to: {self.output_dir}')


    def transform(self):
        '''Alias for mask() — mirrors the MASK class interface.'''
        self.mask()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # if len(sys.argv) < 5:
    #     print(__doc__)
    #     sys.exit(
    #         'Usage: python3 abcd_mask.py <image_dir> <mask_dir> <output_dir> <skip_f> [offset]'
    #     )

    # masker = ABCD_MASK(
    #     image_dir  = sys.argv[1],
    #     mask_dir   = sys.argv[2],
    #     output_dir = sys.argv[3],
    #     skip_f     = int(sys.argv[4]),
    #     offset     = int(sys.argv[5]) if len(sys.argv) > 5 else 0,
    # )

    masker = ABCD_MASK(
        image_dir  = 'C11659/L1C',
        mask_dir   = 'C11659/cloud',
        output_dir = 'C11659/wps_inference/abcd_cloud_100',
        save_model = False,
        skip_f = 100
    )

    masker.mask()