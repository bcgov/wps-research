'''
sampling.py

Sampling from the data, that simple.
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
    Sample n=size values between start and end

    If there is nan in the data, using nan mask to see where the nans are

    Sample until we get enough data size.
    '''

    rng = np.random.default_rng(seed)

    idx = np.arange(start, end)

    if nan_mask is not None:

        idx = idx[np.asarray(nan_mask, bool)]

    if not replacement and size > len(idx):

        raise ValueError("Not enough valid indices to sample from.")

    return rng.choice(idx, size=size, replace=replacement)



def row_sampling(
        X,
        size,
        *,
        replacement = False,
        filter_nan = False,
        seed = 42
):
    '''
    Description
    -----------
    Row sampling from a matrix.

    
    Parameters
    ----------
    X: a 2D or 3D matrix.
    
    size: how many rows to sample.

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

        nan_mask, X = ignore_nan_2D(
            X = X, 
            axis = 1
        )


    #If filter_nan is applied, we want to sample non-nan rows to get up to the size we need.

    sampled_indices = indices_sampling(
        start = 0,
        end   = nrow,
        size  = size,
        nan_mask = nan_mask,
        replacement = replacement,
        seed  = seed
    )


    X_sample = X[sampled_indices]

    return sampled_indices, X_sample










    

