'''
dominant_band.py 

Use this file to quickly see the idea of 'which colour wins'

Syntax:
    python3 dominant_band.py file.bin
'''

from read_raster import Raster

import sys


def dominant_band(
        X, 
        channel
    ):
    '''
    This function imposes a threshold, which compares the selected channel with other 3.

    Parameters
    ----------
    X: the 3 channel dataset

    channel: the channel (band) which you want to compare against the others.

    Returns
    -------
    mask: 2D array (3rd dim is n channel is gone)
    '''
    channel_index = {'r': 0, 'g': 1, 'b': 2}

    chosen_index = channel_index[channel]

    other_indices = [v for v in channel_index.values() if v != chosen_index]

    _, _, n_chan = X.shape

    if n_chan > 3:

        X = X[:, :, :3]

    mask = (
        (X[:, :, chosen_index] > X[:, :, other_indices[0]]) & 

        (X[:, :, chosen_index] > X[:, :, other_indices[1]])
    )

    return mask


def plot_dominant_band(
        X,
        *,
        title = 'RAW',
        figsize = (15, 5)
    ):

    '''
    Applies a logic masking, returns a boolean image where:

    >> White (1): means the chosen channel has the high value at that pixel (dominates)
    >> Black (0): otherwise

    Parameters
    ---------
    X: the dataset to be used
    title: plot title
    figsize: matplotlib size

    Returns
    -------
    A plot of 4 subfigures

    The first one is the input data.

    The other 3 are filtered by rgb channels.
    '''

    import matplotlib.pyplot as plt

    r = dominant_band(X, 'r')
    g = dominant_band(X, 'g')
    b = dominant_band(X, 'b')

    fig, axes = plt.subplots(1, 4, figsize = figsize)

    axes[0].imshow(X)
    axes[0].axis("off")
    axes[0].set_title(title)

    axes[1].imshow(r, cmap='gray')
    axes[1].axis("off")
    axes[1].set_title("red wins")

    axes[2].imshow(g, cmap='gray')
    axes[2].axis("off")
    axes[2].set_title("green wins")

    axes[3].imshow(b, cmap='gray')
    axes[3].axis("off")
    axes[3].set_title("blue wins")

    plt.tight_layout()

    plt.show()
    

if __name__ == '__main__':

    '''
    Use Raster() to read raster and pass the rgb to plot
    '''

    #handling argv
    if len(sys.argv) < 2:
        print("Needs a file name")
        sys.exit(1)

    filename = sys.argv[1]


    #Title
    title = 'RAW'

    if len(sys.argv) > 2: title = sys.argv[2]


    #load raster and read
    raster = Raster(file_name=filename)

    X = raster.readBands_and_trim(crop = True)

    plot_dominant_band(
        X,
        title=title
    )

