'''
nbr.py

Normalized Burned Ratio
'''

from exceptions.matrix_exception import Shape_Mismatched_Error

import numpy as np


def NBR(
        NIR,
        SWIR,
        eps = 1e-3
):
    '''
    Calculates Normalized Burn Ratio for 1 timestamp.

    Parameters
    ----------
    NIR: Near Infrared band

    SWIR: Short-Wave Infrared band

    eps: To prevent Divide By Zero Error


    Returns
    -------
    Values of Normalized Burn Ratio within (-1, 1)
    '''

    try:

        nbr = (NIR - SWIR) / (NIR + SWIR + eps)

    except Exception:

        raise Shape_Mismatched_Error(f"Shape Mismatched, received {NIR.shape} and {SWIR.shape}")
    
    return np.clip(nbr, -1, 1)



def dNBR(
        *,
        NIR_1, SWIR_1,
        NIR_2, SWIR_2,
        eps = 1e-3
):
    '''
    Calculates Differenced Normalized Burn Ratio between 2 timestamps.

    Parameters
    ----------
    NIR_1: Near Infrared band of pre-fire

    SWIR_1: Short-Wave Infrared band of pre-fire

    NIR_2: Near Infrared band of post-fire

    SWIR_2: Short-Wave Infrared band of post-fire

    eps: To prevent Divide By Zero Error


    Returns
    -------
    Values of Normalized Burn Ratio within (-1, 1) for pre and post-fire. With dNBR matrix.
    '''

    nbr_1 = NBR(NIR_1, SWIR_1, eps)

    nbr_2 = NBR(NIR_2, SWIR_2, eps)

    try:

        dnbr = nbr_1 - nbr_2

    except Exception:

        raise Shape_Mismatched_Error(f'NBRs of different shapes, cannot broadcast.')
    
    return nbr_1, nbr_2, dnbr



def plot_barc(
        dNBR, 
        *,
        start_date, 
        end_date,
        figsize=(5, 5),
        fontsize=10
):
    '''
    Plots the BARC 256 burn severity of the provided dNBR and saves it as a png

    Parameters
    ----------
    dNBR: differenced normalized burn ratio data.

    start_date: the day of pre-fire

    end_date: the day of post_fire

    figsize: figure size

    fontsize: legend font size


    Returns
    -------
    A plot of classified burned area reflectance
    '''

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # scale dNBR
    scaled = (dNBR*1000+275)/5 #scalling dNBR

    class_plot = np.full(scaled.shape, np.nan)

    class_plot[scaled < 76] = 1
    class_plot[(scaled >= 76) & (scaled < 110)] = 2
    class_plot[(scaled >= 100) & (scaled < 187)] = 3
    class_plot[scaled >= 187] = 4


    # percentage
    vals, counts = np.unique(class_plot[~np.isnan(class_plot)], return_counts = True)
    perc = dict(zip(vals, np.round(100 * counts / counts.sum(), 1)))

    # plotting
    cmap = ListedColormap(['green', 'yellow', 'orange', 'red'])

    plt.figure(figsize=figsize)
    plt.imshow(class_plot, vmin=1, vmax=4, cmap=cmap)
    plt.title(f'BARC 256 burn severity, start date:{start_date}, end date:{end_date}')

    labels = {
        1: ('Unburned', 'green'),
        2: ('Low', 'yellow'),
        3: ('Medium', 'orange'),
        4: ('High', 'red')
    }

    for k, (name, color) in labels.items():
        plt.scatter(np.nan, np.nan, s=10, marker='s',
                    label=f'{name} {perc.get(k, 0)}%', color=color)
        

    plt.legend(fontsize=fontsize)
    plt.tight_layout()