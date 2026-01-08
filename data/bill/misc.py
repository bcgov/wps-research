import numpy as np


#HISTOGRAM TRIMMING USING PERCENTILE

def htrim(
        X, 
        p=1.0
    ):
    """
    Percentile-based contrast stretch.
    p = lower/upper percentile to clip (e.g. 1.0 -> 1% / 99%)
    
    -> For 1D array
    """
    
    X = X.astype(np.float32)

    lo, hi = np.nanpercentile(X, [p, 100 - p])
    return (X - lo) / (hi - lo)


def htrim_rgb(
        X, 
        p=1.0
    ):
    """
    Percentile-based contrast stretch.
    p = lower/upper percentile to clip (e.g. 1.0 -> 1% / 99%)
    
    -> For 3D array
    """

    X = X.astype(np.float32)

    lo = np.nanpercentile(X, p, axis=(0, 1))
    hi = np.nanpercentile(X, 100 - p, axis=(0, 1))

    return (X - lo) / (hi - lo)



#CROPPING

def crop(
    X,
    tol = 0
):
    
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



    
