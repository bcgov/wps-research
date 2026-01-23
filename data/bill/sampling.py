'''
sampling.py

Designed specifically for inside and outside of polygon.
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

    from misc.general import ignore_nan_2D


    X = X.reshape(-1, X.shape[-1])

    nrow, _ = X.shape #Contains rows with nan (if there is nan in the data)

    nan_mask = None

    if ( filter_nan ):
        #Remove all nan related in rows before sampling

        nan_mask, _ = ignore_nan_2D(
            X = X, 
            axis = 1 #row filter
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

    #masking
    sampled_original_indices = sampled_indices if ( original_indices is None ) else original_indices[sampled_indices]

    X_sample = X[sampled_indices]


    return sampled_original_indices, X_sample



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


    Parameters
    ----------


    Returns
    -------
    '''
    from polygon import split_in_out


    #Extract inside and outside matrix (split)
    inside, in_indices, outside, out_indices = split_in_out(
        raster_dat=raster_dat,
        polygon_dat=polygon_dat,
        raster_filename = raster_filename,
        polygon_filename= polygon_filename
    )

    #Ratio to maintain the true population proportion
    out_in_ratio = len(outside) / len(inside)


    #Sample size (reserves true proportion)
    out_sample_size = int(in_sample_size * out_in_ratio)


    #Sample now
    in_idx_samples,  inside_samples  = row_sampling(X=inside,  
                                                 size=in_sample_size, 
                                                 original_indices=in_indices,
                                                 filter_nan=True,
                                                 seed=seed)
    
    out_idx_samples, outside_samples = row_sampling(X=outside, 
                                                 size=out_sample_size, 
                                                 original_indices=out_indices,
                                                 filter_nan=True,
                                                 seed=seed)
    

    samples = np.vstack([
        inside_samples,
        outside_samples
    ])


    original_indices = np.concatenate([in_idx_samples, 
                                       out_idx_samples])
    

    return original_indices, samples, out_in_ratio









    

