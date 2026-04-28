'''
Sampling data.
'''

import numpy as np


def indices_sampling(
        start: int,
        end: int,
        size: int,
        *,
        nan_mask = None,
        replacement = False,
        seed = 42
):
    '''
    Description
    -----------
    Sample n=size values between start and end - 1.

    If there is nan in the data, using nan mask to see where the nans are

    Sample until we get enough data size.
    '''

    rng = np.random.default_rng(seed)

    idx = np.arange(start, end)

    if nan_mask is not None:

        idx = idx[np.asarray(nan_mask, bool)]

    if not replacement and size > len(idx):

        raise ValueError("(Sampling without replacement): Not enough valid indices to sample from.")

    return rng.choice(idx, size=size, replace=replacement)



def row_sampling(
        X,
        size,
        *,
        original_indices = None,
        replacement = False,
        filter_nan = False,
        seed = 42
):
    '''
    Description
    -----------
    Row sampling from a matrix. A 3D matrix is given, it will reshape to 2D (so each band is now 1 feature).

    Sample from the rows. Designed specifically for pixel sampling.

    
    Parameters
    ----------
    X: a 2D or 3D matrix.
    
    size: how many rows to sample.

    original_indices: sample from the given indices.

    replacement: sampling with replacement or not.
    
    filter nan: To sample without nan values, set to True.

    
    Returns
    -------
    sampled_indices: original index of each sample

    X_sample: sampled data
    '''

    from misc import ignore_nan_2D


    X = X.reshape(-1, X.shape[-1])

    #Contains rows with nan (if there is nan in the data)
    nrow, _ = X.shape
    
    #nan mask is the list of original indices that contain NAN data.
    nan_mask = None

    if ( filter_nan ):
        #Remove all nan related in rows before sampling

        nan_mask, _ = ignore_nan_2D(
            X = X, 

            #axis = 1 is row filter, axis = 0 is column filter.
            axis = 1
        )


    #If filter_nan is applied, we want to sample non-nan rows to get up to the size we need.
    sampled_indices = indices_sampling(
        start = 0,
        end = nrow,
        size  = size,
        nan_mask = nan_mask,
        replacement = replacement,
        seed  = seed
    )

    sampled_original_indices = sampled_indices if ( original_indices is None ) else original_indices[sampled_indices]

    X_sample = X[sampled_indices]

    return sampled_original_indices, X_sample



def stratified_sampling(
        raster_dat,
        hint_dat,
        sample_size: int,
        *,
        target_inside_ratio: float = 0.5,
        seed: int = 42,
):
    '''
    Description
    -----------
    Stratified random sampling on a binary hint mask.

    Aims for ``target_inside_ratio`` of samples inside-hint and the rest
    outside-hint, drawing without replacement and skipping NaN rows. Any
    deficit in one stratum (e.g. tiny fire with few hint pixels) is
    redistributed to the other stratum so the total returned count
    matches ``sample_size`` whenever possible.

    Falls back to uniform random across all valid pixels when one
    stratum has fewer than 5 valid pixels (uninformative hint).

    Parameters
    ----------
    raster_dat : ndarray, shape (H, W, B)
        Used to detect NaN rows.
    hint_dat : ndarray, shape (H, W), bool-like
        The hint mask (True = inside hint).
    sample_size : int
        Total samples to draw.
    target_inside_ratio : float in (0, 1)
        Fraction of ``sample_size`` to draw from inside-hint pixels.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    sampled_indices : ndarray of int
        Flat indices into ``raster_dat.reshape(-1, B)``.
    samples : ndarray, shape (n_drawn, B)
    '''
    flat = raster_dat.reshape(-1, raster_dat.shape[-1])
    valid = np.isfinite(flat).all(axis=1)

    hint_flat = np.asarray(hint_dat, dtype=bool).ravel()
    inside = valid & hint_flat
    outside = valid & ~hint_flat
    n_inside = int(inside.sum())
    n_outside = int(outside.sum())

    rng = np.random.default_rng(seed)

    # Uninformative hint → uniform random over valid pixels (no
    # stratification possible).
    if n_inside < 5 or n_outside < 5:
        valid_idx = np.where(valid)[0]
        n_take = min(sample_size, valid_idx.size)
        if n_take == 0:
            raise ValueError(
                'stratified_sampling: no valid (non-NaN) pixels to sample.')
        sample_indices = rng.choice(valid_idx, size=n_take, replace=False)
        return sample_indices, flat[sample_indices]

    n_in_target = int(round(sample_size * target_inside_ratio))
    n_out_target = sample_size - n_in_target

    n_in_actual = min(n_in_target, n_inside)
    n_out_actual = min(n_out_target, n_outside)

    # Reallocate any deficit so total stays at sample_size when feasible.
    deficit = sample_size - n_in_actual - n_out_actual
    if deficit > 0 and n_in_actual < n_inside:
        extra = min(deficit, n_inside - n_in_actual)
        n_in_actual += extra
        deficit -= extra
    if deficit > 0 and n_out_actual < n_outside:
        extra = min(deficit, n_outside - n_out_actual)
        n_out_actual += extra

    inside_idx = np.where(inside)[0]
    outside_idx = np.where(outside)[0]

    sampled_in = (rng.choice(inside_idx, size=n_in_actual, replace=False)
                  if n_in_actual > 0 else np.array([], dtype=np.int64))
    sampled_out = (rng.choice(outside_idx, size=n_out_actual, replace=False)
                   if n_out_actual > 0 else np.array([], dtype=np.int64))

    sample_indices = np.concatenate([sampled_in, sampled_out])
    rng.shuffle(sample_indices)
    samples = flat[sample_indices]

    return sample_indices, samples


def regular_sampling(
        raster_dat = None,
        *,
        raster_filename: str = None,
        sample_size: int,
        replacement: bool = False,
        seed: int = 42
):
    '''
    Description
    -----------
    Sampling data.

    Notes
    -----
    NO RATIO of any kind is considered.
    '''
    from raster import Raster

    if raster_dat is None and raster_filename is None:

        raise ValueError('You need at least raster data or its filename to process.')
    
    #Assume raster data is always passed in, but raster file_name is always of higher priority.
    if raster_filename is not None:
        
        raster_dat = Raster(raster_filename).read_bands('all')


    #All data have been ready. Now sample.

    org_idx, samples  = row_sampling(
                            X=raster_dat,  
                            size=sample_size,
                            filter_nan=True,
                            replacement=replacement,
                            seed=seed
                        )
    

    return org_idx, samples
    


def in_out_sampling(
        raster_dat=None,
        polygon_dat=None,
        *,
        raster_filename=None,
        polygon_filename=None,
        in_sample_size,
        seed = 42
):
    '''
    Description
    -----------
    Sampling data inside and outside of the polygon.


    Returns
    -------
    original_indices: sampled indices from the original data.
    
    samples: The samples indexed by 'original indices' from the original data.
    
    out_in_ratio: ratio between original data from outside and inside of the polygon.

    
    Notes
    -----
    The sampling will guarantee the ratio of in-out.
    '''

    from polygon import split_in_out


    #Extract inside and outside matrix (split)
    inside_dat, in_indices, outside_dat, out_indices = split_in_out(
        raster_dat=raster_dat,
        polygon_dat=polygon_dat,
        raster_filename = raster_filename,
        polygon_filename= polygon_filename
    )

    #Ratio to maintain the true population proportion
    out_in_ratio = len(inside_dat) / len(outside_dat)


    #Sample size (reserves true proportion)
    out_sample_size = int(in_sample_size * out_in_ratio)


    #Sample now
    org_in_idx, inside_samples  = row_sampling(X=inside_dat,  
                                                 size=in_sample_size, 
                                                 original_indices=in_indices,
                                                 filter_nan=True,
                                                 seed=seed)
    
    org_out_idx, outside_samples = row_sampling(X=outside_dat, 
                                                 size=out_sample_size, 
                                                 original_indices=out_indices,
                                                 filter_nan=True,
                                                 seed=seed)
    

    samples = np.vstack([
        inside_samples,
        outside_samples
    ])


    original_indices = np.concatenate([org_in_idx, 
                                       org_out_idx])
    

    return original_indices, samples, out_in_ratio









    

