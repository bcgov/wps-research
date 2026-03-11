'''
Syntax for quick plot:
  >> python3 change_detection.py file_pre.bin file_post.bin
'''

import numpy as np

import sys

from raster import Raster

from misc import htrim_3d

from plot_tools import plot_multiple


def change_detection(
        pre_X, 
        post_X,
        eps = 1e-3
):
    
    '''
    Description
    -----------
    Compare pixel value between post and pre fire, then normalize it so that it stays between -1 and 1 (theoretically if all data > 0)

        + If the intensity increases in terms of band values (closer to 255), it will be brighter. Otherwise, it will be dark.

        + If the intensity decreases, it will be dark (darker than no changes)

        + If the intensity stays the same, it will be approx. 0


    Parameters
    ----------
    pre_X: pre-fire data

    post_X: post_fire data

    eps: to prevent divide by zero

    
    Returns
    -------
    A differenced Band Normalization.


    Notes
    -----
    Order of images matters.
    '''

    im1 = pre_X.astype(np.float32)
    im2 = post_X.astype(np.float32)

    try:

        d = im2 - im1
        n = im2 + im1

    except Exception:

        raise ValueError(f'Receiving shape {im1.shape} & {im2.shape}')
    
    norm_diff = d / (n + eps)

    return norm_diff



if __name__ == '__main__':

    '''
    Only works if your data has at least 3 channels, will fix.
    '''

    #handling argv
    if len(sys.argv) < 3:
        print("Needs 2 files (pre and post)")
        sys.exit(1)

    filename_pre = sys.argv[1]
    filename_pst = sys.argv[2]

    #load raster and read
    raster_pre_Instance = Raster(file_name=filename_pre)
    raster_pst_Instance = Raster(file_name=filename_pst)

    raster_pre = raster_pre_Instance.readBands_and_trim(
        band_lst=[1,2,3]
    )
    
    raster_pst = raster_pst_Instance.readBands_and_trim(
        band_lst=[1,2,3]
    )

    #Plot title
    title_pre, title_pst = raster_pre_Instance.acquisition_timestamp, raster_pst_Instance.acquisition_timestamp

    htrim_nbd = htrim_3d(
        change_detection(pre_X=raster_pre, 
                        post_X=raster_pst)
    )

    #plot result
    plot_multiple(
        X_list = [raster_pre, raster_pst, htrim_nbd],
        title_list = [f'Pre: {title_pre}', 
                      f'Post: {title_pst}',
                      'Changes - used first 3 bands (contrast streching applied)'],
        max_per_row=3
    )