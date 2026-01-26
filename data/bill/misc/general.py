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
    Description
    -----------
    Percentile-based contrast stretch.

    p = lower/upper percentile to clip (e.g. 1.0 -> 1% / 99%)
    
    This is for 1D arrays.


    Parameters
    ----------
    X: The data to be used.
    
    p: The cumulative percentage for each tail.


    Returns
    -------
    Still a 1D array but trimmed.
    '''
    
    X = X.astype(np.float32)

    lo, hi = np.nanpercentile(X, [p, 100 - p])

    return (X - lo) / (hi - lo)


def htrim_3d(
        X, 
        p=1.0
    ):
    '''
    Description
    -----------
    htrim on each of the channels independently.

    
    Parameters
    ----------
    X: The data to be used.
    
    p: The cumulative percentage for each tail.


    Returns
    -------
    Still a 3D matrix but trimmed.
    '''

    X = X.astype(np.float32)

    lo, hi = np.nanpercentile(X, p, axis=(0, 1)), np.nanpercentile(X, 100 - p, axis=(0, 1))

    return (X - lo) / (hi - lo)



def crop_no_data(
        X,
        tol = 0
):
    '''
    Description
    -----------
    Raster data can have the no-data parts. We want to crop them out, automatically.


    Parameters
    ----------
    X: matrix to crop

    tol: tolerance, condition to crop


    Returns
    -------
    A cropped matrix.


    Notes
    -----
    Only works for 3D arrays for now.

    This function is pretty tricky. Use with caution.

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
    X1: 1st matrix (3D)
    
    X2: 2nd matrix (3D)


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
    X: a 2D matrix.

    axis: 0 is column, 1 is row


    Returns
    -------
    mask: TRUE means not nan, FALSE means nan

    filtered: A filtered 2D matrix.
    '''

    mask = ~np.isnan(X).any(axis=axis)

    filtered = X[mask]

    return mask, filtered



def extract_border(
        mask: np.ndarray,
        thickness: int = 1
):
    
    '''
    Description
    -----------
    Extracts the border of a whole polygon area.


    Parameters
    ----------
    A 2D array of boolean dtype

    E.g: Everything outside of the polygon is False and True otherwise

    
    Returns
    -------
    The same thing but inside will be false. Leaving us just the border.
    '''
    from scipy.ndimage import binary_erosion

    if thickness < 1:
        raise ValueError("thickness must be >= 1")

    mask = mask.astype(bool)

    eroded = binary_erosion(
        mask,
        structure=np.ones((3, 3)),
        iterations=thickness
    )

    border = mask & (~eroded)
    return border



def draw_border(
    img: np.ndarray,
    border: np.ndarray,
    color=(255, 0, 0)
):
    """
    Description
    -----------
    Add border and colour it to a raster image.


    Parameters
    ----------
    img    : (H, W, C)

    border : (H, W) boolean

    color  : length-C tuple - Default is RED.


    Returns
    -------
    Raster image with coloured polygon applied directly onto it.
    """

    from exceptions.matrix import Shape_Mismatched_Error

    try:
        assert border.shape == img.shape[:2]

    except Exception:

        raise Shape_Mismatched_Error("Image and border don't share the same 2D shape.")

    out = img.copy()

    out[border] = color

    return out




def is_boolean_matrix(
        A
):
    '''
    Description
    -----------
    Checks if data are of boolean dtype.
    '''

    A = np.asarray(A)

    # Case 1: true boolean dtype
    if A.dtype == np.bool_:
        return True

    # Case 2: numeric but only 0/1
    if np.issubdtype(A.dtype, np.integer):
        return np.all((A == 0) | (A == 1))

    if np.issubdtype(A.dtype, np.floating):
        return np.all((A == 0.0) | (A == 1.0))

    return False




def get_combinations(
        val_lst: list,
        least: int = 1,
        most: int = None
):

    '''
    Description
    -----------
    Get combination of a list (order doesn't matter)


    Parameters
    ----------
    val_lst: list of values to find comb

    least: at least how many values in each combination.

    most: at least how many values in each combination.


    Returns
    -------
    list[list]
    '''

    from itertools import combinations


    n = len(val_lst)

    if most is None:
        most = n

    if least < 1 or most > n or least > most:
        raise ValueError("Invalid least/most values")

    return [
        list(c)
        for k in range(least, most + 1)
        for c in combinations(val_lst, k)
    ]
