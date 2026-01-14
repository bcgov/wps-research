'''
misc/general.py

You can call it a miscellaneous file, the functions can be useful for lots of tasks across the modules.
'''

import numpy as np



##### HISTOGRAM TRIMMING USING PERCENTILE #####

def htrim_1d(
        X, 
        p=1.0
    ):
    '''
    Trim 'outlier' pixel values for visualization purpose.

    Percentile-based contrast stretch.

    p = lower/upper percentile to clip (e.g. 1.0 -> 1% / 99%)
    
    This is for 1D arrays.

    Parameters
    ----------
    X: The data to be used.
    
    p: The cumulative percentage for each tail.
    '''
    
    X = X.astype(np.float32)

    lo, hi = np.nanpercentile(X, [p, 100 - p])

    return (X - lo) / (hi - lo)


def htrim_3d(
        X, 
        p=1.0
    ):
    '''
    Refer to htrim_1d
    
    This is for 3D arrays.

    Parameters
    ----------
    X: The data to be used.
    
    p: The cumulative percentage for each tail.
    '''

    X = X.astype(np.float32)

    lo, hi = np.nanpercentile(X, p, axis=(0, 1)), np.nanpercentile(X, 100 - p, axis=(0, 1))

    return (X - lo) / (hi - lo)



##### CROPPING #####

def crop_no_data(
        X,
        tol = 0
):
    '''
    Parameters
    ----------
    X: matrix to crop

    tol: tolerance, condition to crop


    Returns
    -------
    A cropped matrix


    Notes
    -----
    If applied to each of more than 1 matrix, may want to use match_shape() to balance the shape for arithmetric broadcasting.
    '''
    
    H, W, _ = X.shape

    top = 0
    while top < H:
        r = X[top, :, :]
        if abs(r.max() - r.min()) <= tol:
            top += 1
        else:
            break

    bottom = H
    while bottom > top:
        r = X[bottom - 1, :, :]
        if abs(r.max() - r.min()) <= tol:
            bottom -= 1
        else:
            break

    left = 0
    while left < W:
        c = X[:, left, :]
        if abs(c.max() - c.min()) <= tol:
            left += 1
        else:
            break

    right = W
    while right > left:
        c = X[:, right - 1, :]
        if abs(c.max() - c.min()) <= tol:
            right -= 1
        else:
            break

    return X[top:bottom, left:right, :]



##### Array adjustment #####

def match_shape(
        X1,
        X2
):
    '''
    Crop rows, cols in a way that it will match the shape of the files.

    In this version, it will crop from right to left, bottom to top.

    
    Parameters
    ----------
    X1: 1st matrix (multi dim)
    
    X2: 2nd matrix (multi dim)


    Returns
    -------
    X1_cropped, X2_cropped. They match shape. 

    
    Notes
    -----
    This doesn't affect the number of channels.
    Should use just for Visualization when the crop happens lightly.
    '''

    r1, c1, _ = X1.shape
    r2, c2, _ = X2.shape

    r_min = min(r1, r2)
    c_min = min(c1, c2)

    X1_cropped = X1[:r_min, :c_min, :]
    X2_cropped = X2[:r_min, :c_min, :]

    return X1_cropped, X2_cropped



#Nan values

def ignore_nan_2D(
        X,
        axis = 1
):
    '''
    Filters out rows / cols with nan. We also want to know, after filtering, 

    what is the orignal row index that a row in the filtered data correponds to. 


    Parameters
    ----------
    X: a 2D matrix

    axis: 0 is column, 1 is row


    Returns
    -------
    mask: TRUE means not nan, FALSE means nan

    filtered: A filtered 2D matrix.
    '''

    mask = ~np.isnan(X).any(axis=axis)

    filtered = X[mask]

    return mask, filtered





    
