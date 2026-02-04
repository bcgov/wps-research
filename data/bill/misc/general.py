'''
miscellanous file for general purposes.
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

    try:
        assert border.shape == img.shape[:2]

    except Exception:

        raise ValueError("Image and border don't share the same 2D shape.")

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
