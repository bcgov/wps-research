'''
Please refer to README.md as the intructions.
'''

import os
from dataclasses import dataclass
from raster import Raster
import numpy as np
from datetime import datetime
from pathlib import Path


#######################
#       Utilities     #
#######################

def extract_datetime(
        filename: str
) -> datetime:
    
    '''
    datetime in ISO 8601 in UTC
    '''
    
    raster = Raster(filename)

    try:
        acquisition_time = raster.acquisition_time
        dt = datetime.fromisoformat(acquisition_time.replace("Z", ""))
        return dt.replace(microsecond=0, tzinfo=None)

    except Exception:
        raise ValueError(f"Cannot extract acquisition timestamp from \n{filename}")
    

def get_ordered_file_dict(
        image_dir,
        mask_dir=None,
        *,
        start: datetime = None,
        end: datetime = None
):
    from misc import iter_files
    dictionary = {}

    # Normalize to list
    image_dirs = [image_dir] if isinstance(image_dir, str) else list(image_dir)
    # Collect image files
    for i_dir in image_dirs:
        for img_f in iter_files(i_dir, '.bin'):
            acquisition_time = extract_datetime(img_f)
            if (start is not None and acquisition_time < start) or \
               (end is not None and acquisition_time > end):
                continue
            dictionary[acquisition_time] = dictionary.get(acquisition_time, {})
            dictionary[acquisition_time].setdefault('image_path', []).append(img_f)

    # Remove entries without all image dirs represented
    dictionary = {
        date: file_dict
        for date, file_dict in dictionary.items()
        if len(file_dict.get('image_path', [])) == len(image_dirs)
    }

    dictionary = dict(sorted(dictionary.items()))

    # If no mask_dir → return early
    if mask_dir is None:
        print(f"Dictionary completed | it stores {len(dictionary)} timestamps.")
        return dictionary

    # Normalize to list
    mask_dirs = [mask_dir] if isinstance(mask_dir, str) else list(mask_dir)

    # Collect mask files
    for m_dir in mask_dirs:
        omitted_mask_f = 0
        for mask_f in iter_files(m_dir, '.bin'):
            acquisition_time = extract_datetime(mask_f)
            if acquisition_time in dictionary:
                dictionary[acquisition_time].setdefault('mask_path', []).append(mask_f)

            else:
                print(f'Date = {acquisition_time} from image_dir not in {m_dir}')
                omitted_mask_f += 1
        
        print(f'\n -> Ommited {omitted_mask_f} masks in {m_dir}')
        print('-' * 30)
                
    print(f'Iterating completed | omitted {omitted_mask_f} mask files.')

    # Remove entries without all mask dirs represented
    dictionary = {
        date: file_dict
        for date, file_dict in dictionary.items()
        if len(file_dict.get('mask_path', [])) == len(mask_dirs)
    }

    # Unwrap single-item lists back to plain strings
    for file_dict in dictionary.values():
        if isinstance(image_dir, str):
            file_dict['image_path'] = file_dict['image_path'][0]
        if isinstance(mask_dir, str):
            file_dict['mask_path'] = file_dict['mask_path'][0]

    print(f"Dictionary completed | it stores {len(dictionary)} timestamps.")
    return dictionary



def writeENVI(
    output_filename: str,
    data: np.ndarray,
    *,
    ref_filename: str | None = None,
    band_names: list[str] | None = None,
    copy_geo: bool = True,
    same_hdr: bool = False,
    nodata_val: float | None = np.nan
):
    """
    - creates a brand-new ENVI file
    - ignores pixel values in ref_filename
    - optionally copies georeferencing
    """

    from osgeo import gdal
    import shutil

    if data.ndim == 2:
        data = data[..., None]

    H, W, K = data.shape

    if band_names is not None and len(band_names) != K:
        raise ValueError("band_names length mismatch")

    driver = gdal.GetDriverByName("ENVI")

    if ref_filename is None:
        raise ValueError("ref_filename required to define geometry")

    ref = gdal.Open(ref_filename)
    if ref is None:
        raise RuntimeError("Failed to open reference file")

    out = driver.Create(
        output_filename,
        W, H,
        K,
        gdal.GDT_Float32
    )

    if copy_geo:
        out.SetGeoTransform(ref.GetGeoTransform())
        out.SetProjection(ref.GetProjection())

    for i in range(K):
        band = out.GetRasterBand(i + 1)
        band.WriteArray(data[..., i])

        if nodata_val is not None:
            band.SetNoDataValue(float(nodata_val))
            
        if band_names:
            band.SetDescription(band_names[i])

    out.FlushCache()
    out = ref = None

    if same_hdr:
        ref_hdr = Path(ref_filename).with_suffix(".hdr")
        out_hdr = Path(output_filename).with_suffix(".hdr")
        if ref_hdr.exists():
            shutil.copy(ref_hdr, out_hdr)

        else:
            print(ref_hdr)
            
    return


#######################
#       Base Class    #
#######################

@dataclass
class LookBack:

    #Files
    image_dir: str
    mask_dir: str = None
    output_dir: str = None

    #Basic settings
    mask_threshold: float = 1e-5
    max_lookback_days: int = 30
    nodata_val = np.nan

    #Date selections
    start: datetime = None
    end: datetime = None

    #Miscellaneous
    png: bool = True

    #Parallel processing
    n_workers: int = 8


    def get_file_dictionary(
            self
    ):

        file_dict = get_ordered_file_dict(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            start=self.start,
            end=self.end
        )

        if len(file_dict) == 0:

            raise ValueError('\nEmpty satisfying files, cannot process\nCheck start and end dates inputs.')
        
        return file_dict
    


    def get_file_path(
            self,
            date: datetime,
            key: str
    ):
        
        return self.file_dict[date][key]
    


    def read_image(
            self, 
            date: datetime,
            band_list: list[int] = None
    ):
        
        '''
        Read data from image.
        Combines multiple images if image_path is a list.
        '''
        image_paths = self.get_file_path(date, 'image_path')

        if isinstance(image_paths, str):
            return Raster(image_paths).read_bands('all' if band_list is None else [b-1 for b in band_list])

        return [Raster(img_f).read_bands(band_list) for img_f in image_paths]
    
    

    def read_mask(
            self,
            date: datetime,
            as_prob: bool = False
    ):
        '''
        Read mask on that date.
        Combines multiple masks if mask_path is a list.
        Also returns coverage of the mask.

        If as_prob=True:
            - Returns list of arrays if multiple masks, single array if one mask
            - Binary masks (0/1) are returned as bool even with as_prob=True
            - Probabilistic masks are returned as float in [0, 1]
            - Coverage is returned as None when returning a list
        If as_prob=False:
            - All masks are combined via logical OR into a single bool mask
            - Returns (mask, coverage)
        '''
        mask_paths = self.get_file_path(date, 'mask_path')
        single     = isinstance(mask_paths, str)
        if single:
            mask_paths = [mask_paths]

        processed = []
        for mask_f in mask_paths:
            raw = Raster(mask_f).read_bands().squeeze()

            max_val = raw.max()

            if max_val <= 1.0:
                # Already in [0, 1] or binary — no scaling needed
                processed.append(raw.astype(np.float32))
            else:
                # Assumed to be in [0, 100] — scale down
                processed.append((raw / 100.).astype(np.float32))

        if as_prob:
            if single:
                return processed[0], float(processed[0].mean())
            return processed, None

        # Combine into a single boolean mask
        bool_masks = []
        for data in processed:
            if self.mask_threshold is None:
                bool_masks.append(data.astype(np.bool_))
            else:
                bool_masks.append(data >= self.mask_threshold)

        mask     = np.logical_or.reduce(bool_masks)
        coverage = float(mask.mean())
        return mask, coverage
    

    def get_nodata_mask(
            self,
            img_dat: np.ndarray,
            nodata_val: float = np.nan
    ) -> np.ndarray:
        
        '''
        not recommended, not matches true no data mask 100%
        '''
        
        if np.isnan(nodata_val):
            return np.all(np.isnan(img_dat), axis=-1)
        else:
            return np.all(img_dat == nodata_val, axis=-1)


#######################
#       Cloud Masking #
#######################

'''abcd_rf.py: "A is to B as C is to D" using Random Forest regression.
Pixel-based analogy: trains RF on (A, B) pairs, applies to C to produce D.
Supports ENVI (.bin/.hdr) and GeoTIFF (.tif) input. Output is ENVI format.
Uses GDAL for raster I/O. Caches trained RF model as .pkl.
Usage: python3 abcd_rf.py A B C skip_f [offset]
A - input image (n bands), training features
B - input image (m bands), training targets
C - input image (n bands), inference features
skip_f - spatial sampling stride for training pixels
offset - spatial sampling offset (default 0)
Output D is written as ENVI .bin/.hdr with CRS/map info copied from C.
'''
import sys
import os
import pickle
import numpy as np
try:
    from osgeo import gdal
except ImportError:
    import gdal

gdal.AllRegister()
gdal.UseExceptions()

def read_raster(path):
    '''Read a raster file (ENVI or TIFF) via GDAL. Returns (data [bands, pixels], ds).'''
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        sys.exit(f"Error: cannot open {path}")
    nb = ds.RasterCount
    nr = ds.RasterYSize
    nc = ds.RasterXSize
    npx = nr * nc
    data = np.empty((nb, npx), dtype=np.float32)
    for b in range(nb):
        band = ds.GetRasterBand(b + 1)
        arr = band.ReadAsArray().astype(np.float32).ravel()
        data[b, :] = arr
    return data, ds


def is_bad(data, i):
    '''Check if pixel i is bad (NaN, Inf, or all-zero for multi-band).'''
    vals = data[:, i]
    if np.any(np.isnan(vals)) or np.any(np.isinf(vals)):
        return True
    nb = data.shape[0]
    if nb > 1 and np.all(vals == 0):
        return True
    return False


def write_envi(outpath, data_2d, ref_ds, nb_out):
    '''Write ENVI BSQ output (.bin + .hdr) with map/CRS info from ref_ds.'''
    nr = ref_ds.RasterYSize
    nc = ref_ds.RasterXSize
    driver = gdal.GetDriverByName('ENVI')
    out_ds = driver.Create(outpath, nc, nr, nb_out, gdal.GDT_Float32)
    if out_ds is None:
        sys.exit(f"Error: cannot create {outpath}")
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    for b in range(nb_out):
        out_band = out_ds.GetRasterBand(b + 1)
        arr = data_2d[b, :].reshape(nr, nc)
        out_band.WriteArray(arr)
        out_band.SetNoDataValue(float('nan'))
    out_ds.FlushCache()
    out_ds = None

def cache_key(path_a, path_b, skip_f, offset):
    '''Build cache filename from basenames of A, B plus skip_f and offset.'''
    a_base = os.path.splitext(os.path.basename(path_a))[0]
    b_base = os.path.splitext(os.path.basename(path_b))[0]
    return f"{a_base}_{b_base}_{skip_f}_{offset}.pkl"


# ---------------------------------------------------------------------------
# Array-level core — used by abcd_mask.py and any other callers that already
# have data in memory.  No file I/O here, no caching.
# ---------------------------------------------------------------------------

def bad_pixel_mask_arrays(data: np.ndarray) -> np.ndarray:
    '''
    Boolean (n_pixels,) — True where pixel should be excluded.
    Excluded if any band is NaN/Inf, or (multi-band only) all bands are zero.
    '''
    bp = np.any(~np.isfinite(data), axis=0)
    if data.shape[0] > 1:
        bp |= np.all(data == 0, axis=0)
    return bp


def fit_rf(X_train: np.ndarray, Y_train: np.ndarray):
    '''
    Fit a RandomForestRegressor on pre-built (X, Y) arrays.

    Parameters
    ----------
    X_train : (n_samples, n_features)
    Y_train : (n_samples,) or (n_samples, n_targets)

    Returns
    -------
    Trained RF model.
    '''
    print("Fitting RandomForestRegressor ...")
    try:
        from cuml.ensemble import RandomForestRegressor
        print("  (using cuML GPU implementation)")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42
        )

    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        print("  (cuML not available — falling back to sklearn)")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
    rf.fit(X_train, Y_train)
    return rf



def predict_rf(rf, C: np.ndarray, bad_pixels: np.ndarray = None) -> np.ndarray:
    '''
    Run inference with a trained RF on array C.

    Parameters
    ----------
    rf         : trained RF model
    C          : (n_bands, n_pixels) float32 — inference features
    bad_pixels : (n_pixels,) bool — pixels to skip (NaN in output).
                 If None, computed automatically from C.

    Returns
    -------
    D : (n_targets, n_pixels) float32 — predictions, NaN at bad pixels.
        If the model has a single output, shape is (1, n_pixels).
    '''
    n_px = C.shape[1]

    if bad_pixels is None:
        bad_pixels = bad_pixel_mask_arrays(C)

    good_idx = np.where(~bad_pixels)[0]

    # Determine number of outputs from a tiny probe
    probe    = rf.predict(C[:, good_idx[:1]].T)
    n_out    = 1 if probe.ndim == 1 else probe.shape[1]

    D = np.full((n_out, n_px), np.nan, dtype=np.float32)

    if len(good_idx) == 0:
        return D

    preds = rf.predict(C[:, good_idx].T)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    for b in range(n_out):
        D[b, good_idx] = preds[:, b].astype(np.float32)

    print(f"  predicted {len(good_idx)} pixels")
    return D



def abcd_rf_arrays(
    A:      np.ndarray,
    B:      np.ndarray,
    C:      np.ndarray,
    skip_f: int,
    offset: int = 0,
    pkl_path: str = None,
) -> tuple:
    '''
    Array-level A:B::C:D RF regression.  No file I/O.

    Parameters
    ----------
    A        : (n_bands_a, n_pixels) float32 — training features
    B        : (n_bands_b, n_pixels) float32 — training targets
    C        : (n_bands_a, n_pixels) float32 — inference features
    skip_f   : sampling stride
    offset   : sampling offset (default 0)
    pkl_path : if given, save/load the trained model at this path

    Returns
    -------
    (D, rf)
    D  : (n_bands_b, n_pixels) float32 — predictions
    rf : trained RF model
    '''
    # ---------------------------------------------------------------- model
    if pkl_path and os.path.isfile(pkl_path):
        print(f"Loading cached model: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            rf = pickle.load(f)
    else:
        n_px = A.shape[1]
        if A.shape[1] != B.shape[1]:
            raise ValueError("A and B must have the same number of pixels.")
        if skip_f < 1 or skip_f >= n_px:
            raise ValueError(f"Illegal skip_f={skip_f}.")

        # Bad-pixel mask on A
        bp = bad_pixel_mask_arrays(A)
        print(f"  {np.sum(bp)} bad pixels, {np.sum(~bp)} good pixels in A")

        idx      = np.arange(offset, n_px, skip_f)
        good_idx = idx[~bp[idx]]
        n_train  = len(good_idx)
        print(f"Training samples: {n_train} (skip_f={skip_f}, offset={offset})")

        if n_train == 0:
            raise ValueError("No good training pixels after sampling.")

        X_train = A[:, good_idx].T
        Y_train = B[:, good_idx].T
        np.nan_to_num(Y_train, copy=False, nan=0.0)

        rf = fit_rf(X_train, Y_train)

        if pkl_path:
            print(f"Saving model to {pkl_path}")
            with open(pkl_path, 'wb') as f:
                pickle.dump(rf, f)

    # --------------------------------------------------------------- infer
    D = predict_rf(rf, C)
    return D, rf


# ---------------------------------------------------------------------------
# Original file-based entry point — unchanged
# ---------------------------------------------------------------------------

def abcd_rf(path_a, path_b, path_c, skip_f, offset=0, write_output=False):
    pkl_name = cache_key(path_a, path_b, skip_f, offset)

    if os.path.isfile(pkl_name):
        print(f"Loading cached model: {pkl_name}")
        with open(pkl_name, 'rb') as f:
            rf = pickle.load(f)
        print("Reading C ...")
        C, ds_c = read_raster(path_c)
        nb_b = rf.n_features_out_ if hasattr(rf, 'n_features_out_') else rf.n_outputs_
    else:
        print("Reading A ...")
        A, ds_a = read_raster(path_a)
        print("Reading B ...")
        B, ds_b = read_raster(path_b)
        print("Reading C ...")
        C, ds_c = read_raster(path_c)

        nb_b = B.shape[0]

        print("Flagging bad pixels in A ...")
        bp_ab = np.any(np.isnan(A) | np.isinf(A), axis=0)
        if A.shape[0] > 1:
            bp_ab |= np.all(A == 0, axis=0)
        if bp_ab.all():
            sys.exit("Error: no good pixels in A x B")
        print(f"  {np.sum(bp_ab)} bad pixels, {np.sum(~bp_ab)} good pixels in A")

        sample_idx = np.arange(offset, A.shape[1], skip_f)
        good_mask  = ~bp_ab[sample_idx]
        sample_idx = sample_idx[good_mask]
        n_train    = len(sample_idx)
        print(f"Training samples: {n_train} (skip_f={skip_f}, offset={offset})")
        if n_train == 0:
            sys.exit("Error: no good training pixels after sampling")

        X_train = A[:, sample_idx].T
        Y_train = B[:, sample_idx].T
        np.nan_to_num(Y_train, copy=False, nan=0.0)

        print("Fitting RandomForestRegressor ...")
        try:
            from cuml.ensemble import RandomForestRegressor
            print("  (using cuML GPU implementationssss)")
            rf = RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42)
        except ImportError:
            from sklearn.ensemble import RandomForestRegressor
            print("  (cuML not available — falling back to sklearn)")
            rf = RandomForestRegressor(n_estimators=100, max_depth=None,
                                       random_state=42, n_jobs=-1)
        rf.fit(X_train, Y_train)

        print(f"Saving model to {pkl_name}")
        with open(pkl_name, 'wb') as f:
            pickle.dump(rf, f)

    print("Inference on C ...")
    D = np.full((nb_b, C.shape[1]), np.nan, dtype=np.float32)
    bp_c    = np.any(np.isnan(C) | np.isinf(C), axis=0)
    if C.shape[0] > 1:
        bp_c |= np.all(C == 0, axis=0)
    good_c  = np.where(~bp_c)[0]
    X_infer = C[:, good_c].T
    preds   = rf.predict(X_infer)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    for b in range(nb_b):
        D[b, good_c] = preds[:, b].astype(np.float32)
    print(f"  predicted {len(good_c)} pixels")

    if write_output:
        a_base     = os.path.splitext(os.path.basename(path_a))[0]
        b_base     = os.path.splitext(os.path.basename(path_b))[0]
        c_base     = os.path.splitext(os.path.basename(path_c))[0]
        out_prefix = f"abcd_{a_base}_{b_base}_{c_base}_{skip_f}_{offset}"
        out_bin    = out_prefix + ".bin"
        print(f"Writing {out_bin} ...")
        write_envi(out_bin, D, ds_c, nb_b)
        plot_bands = "1 2 3" if nb_b >= 3 else "1 1 1"
        print("Plotting ...")
        os.system(f"raster_plot.py {out_bin} {plot_bands} 1")
        print("Done.")

    return D



#!/usr/bin/env python3
'''
abcd_mask.py — Per-date "A:B::C:D" cloud masking via Random Forest regression.

Delegates all RF logic to abcd_rf.py (abcd_rf_arrays).

For each date independently:
    A = image         (band values)       — training features
    B = cloud prob    (mask probability)  — training targets
    C = image again   (same as A)         — inference features
    D = predicted cloud probability map   — saved as ENVI .bin/.hdr + .png

Arguments
---------
    image_dir  — directory of satellite images  (acquisition time from metadata)
    mask_dir   — directory of cloud probability masks  (matched by acquisition time)
    output_dir — directory where .bin / .hdr / .png outputs are written
    skip_f     — spatial sampling stride  (e.g. 10 = every 10th pixel)
    offset     — sampling offset, default 0
'''

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from osgeo import gdal
except ImportError:
    import gdal

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


#######################
#       Mask to NAN   #
#######################

from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

@dataclass
class MASK_TO_NODATA(LookBack):

    same_htrim: bool = False

    def __post_init__(self):

        self.file_dict = self.get_file_dictionary()

        # Extract level and grid from first image
        first_image = list(self.file_dict.values())[0]['image_path']

        if isinstance(first_image, list):
            first_image = first_image[0]

        file_fields  = Path(first_image).name.split('_')
        self.level   = file_fields[1][3:5]   # e.g. 'MSIL2A' -> 'L2'
        self.grid    = file_fields[5]      # e.g. 'T09UYU'
        self.output_dir = f"{self.level}_{self.grid}"

        os.makedirs(self.output_dir, exist_ok=True)


    def save_png(
            self, 
            raw, cleaned, 
            image_path):

        fig = Figure(figsize=(38, 20))
        FigureCanvas(fig)
        axes = fig.subplots(1, 2)

        lo = np.nanpercentile(raw, 1, axis=(0, 1))
        hi = np.nanpercentile(raw, 99, axis=(0, 1))

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
            same_hdr=True,
            nodata_val=self.nodata_val
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

    import argparse
    import os
    import subprocess

    parser = argparse.ArgumentParser(
        description="Run cloud masking and then mask-to-nodata cleaning."
    )

    # Positional
    parser.add_argument("image_dir",
                        help="Path to L1C image directory.")
    parser.add_argument("cloud_mask_dir",
                        help="Directory used to train/predict the cloud mask (input mask dir for ABCD_MASK).")

    # Optional
    parser.add_argument("--extra_mask_dirs",  type=str,   default=None,
                        help="Comma-separated extra mask directories passed to MASK_TO_NODATA (e.g. 'C11659/shadow,C11659/nodata').")
    parser.add_argument("--skip_f",           type=int,   default=5555,
                        help="Skip factor for ABCD_MASK (default: 5555).")
    parser.add_argument("--save_model",       action="store_true", default=False,
                        help="Save the trained ABCD model (default: False).")
    parser.add_argument("--mask_threshold",   type=float, default=1e-4,
                        help="Mask threshold for MASK_TO_NODATA (default: 1e-4).")
    parser.add_argument("--run_mask2nan", action="store_true", default=False,
                    help="If set, run MASK_TO_NODATA after cloud masking. Default: False (stop after masking).")
    parser.add_argument("--output_dir_mask",  type=str,   default=None,
                        help="Output dir for cloud mask. Defaults to ./cloud_mask_{skip_f}.")
    parser.add_argument("--predicted_cloud",  type=str,   default=None,
                        help="Path to a pre-computed cloud mask directory. If provided, ABCD_MASK is skipped entirely.")
    parser.add_argument("--n_workers",        type=int,   default=8,
                        help="Number of workers for MASK_TO_NODATA (default: 8).")

    args = parser.parse_args()

    # Parse comma-separated extra mask dirs into a list (empty if not provided)
    extra_mask_dirs = [d.strip() for d in args.extra_mask_dirs.split(",")] if args.extra_mask_dirs else []

    # Resolve output dirs
    output_dir_mask  = args.output_dir_mask  or f"./cloud_mask_{args.skip_f}"

    # ------------------------------------------------------------------ #
    #  Step 1 – Cloud masking (skip if predicted_cloud is supplied)       #
    # ------------------------------------------------------------------ #
    if args.predicted_cloud:
        predicted_cloud_dir = args.predicted_cloud
        print(f"[INFO] Skipping ABCD_MASK. Using pre-computed cloud mask at: "
              f"{os.path.abspath(predicted_cloud_dir)}")
    else:
        print("[INFO] Running ABCD_MASK ...")
        masker = ABCD_MASK(
            image_dir  = args.image_dir,
            mask_dir   = args.cloud_mask_dir,
            output_dir = output_dir_mask,
            save_model = args.save_model,
            skip_f     = args.skip_f,
            end=datetime(2025,5,10)
        )
        masker.mask()
        predicted_cloud_dir = output_dir_mask
        print(f"[INFO] Cloud mask saved to: {os.path.abspath(predicted_cloud_dir)}")

    # ------------------------------------------------------------------ #
    #  Step 2 – User review                                               #
    # ------------------------------------------------------------------ #
    if args.run_mask2nan:
        print(f"\n\n[REVIEW] Please check the cloud mask at: {os.path.abspath(predicted_cloud_dir)}")
        answer = input("Are you satisfied with the mask? Continue to mask2nodata? [Y/n]: ").strip()
        if answer.lower() == "n":
            print("[ABORT] Exiting. Re-run with --predicted_cloud to skip masking.")
            exit(0)

    # ------------------------------------------------------------------ #
    #  Step 3 – Mask to nodata                                            #
    # ------------------------------------------------------------------ #
        print('\n\n')
        print("[INFO] Running MASK_TO_NODATA ...")
        cleaner = MASK_TO_NODATA(
            image_dir      = args.image_dir,
            mask_dir       = [predicted_cloud_dir] + extra_mask_dirs,
            n_workers      = args.n_workers,
            mask_threshold = args.mask_threshold
        )
        output_dir_clean = cleaner.output_dir

        cleaner.run()

        print(f"[INFO] Cleaned data written to: {os.path.abspath(output_dir_clean)}")

        try:
            # ---------------------------------------------------------------- #
            #  Step 4 – MRAP                                                   #
            # ---------------------------------------------------------------- #

            mrap_cmd = ["sentinel2_mrap.py", cleaner.grid]

            print(cleaner.level)
            if cleaner.level == 'L1':
                mrap_cmd.append("--L1")
                
            mrap_cwd = os.path.dirname(os.path.abspath(output_dir_clean)) or "."
            print(f"\n\n[INFO] Running MRAP: {' '.join(mrap_cmd)} (cwd={mrap_cwd})")
            subprocess.run(mrap_cmd, cwd=mrap_cwd, check=True)

        except Exception:

            print('Cannot mrap.')

        
        try:

            # ---------------------------------------------------------------- #
            #  Step 5 – MP4                                                    #
            # ---------------------------------------------------------------- #
            print(f"\n\n[INFO] Running sentinel2_mp4 in {os.path.abspath(output_dir_clean)} ...")
            subprocess.run(["sentinel2_mp4"], cwd=os.path.abspath(output_dir_clean), check=True)
            print("[INFO] All done.")

        except Exception:
            
            print('Cannot generate mp4.')

    else:
        print(f"\n[INFO] Cloud masking complete. Masks saved to: {os.path.abspath(predicted_cloud_dir)}")
        print("[INFO] Re-run with --run_mask2nan to proceed to cleaning.")