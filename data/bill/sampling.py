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
        replacement = False,
        seed = 42
):
    '''
    Sample n=size values between start and end
    '''

    rng = np.random.default_rng(seed=seed)

    idx = rng.choice(np.arange(start, end), size=size, replace=replacement)

    return idx



def row_sampling(
        X,
        size,
        *,
        replacement = False,
        filter_nan = False,
        seed = 42
):
    '''
    Row sampling from a matrix.

    
    Parameters
    ----------
    X: a 2D or 3D matrix.
    
    size: how many rows to sample.

    replacement: sampling with replacement or not.
    
    filter nan: To sample without nan values, set to True.

    
    Returns
    -------
    Sampled rows of size size.
    '''

    from misc.general import ignore_nan_2D


    shape = X.shape

    X = X.reshape(-1, shape[-1])

    if ( filter_nan ):
        #Remove all nan related in rows before sampling

        X = ignore_nan_2D(
            X = X, 
            axis = 1
        )

    new_nrow, _ = X.shape

    sampled_indices = indices_sampling(
        start = 0,
        end   = new_nrow,
        size  = size,
        replacement = replacement,
        seed  = seed
    )

    return X[sampled_indices]










    

