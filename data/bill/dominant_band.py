'''
Use this file to quickly see the idea of 'which colour wins'

python3 dominant_band.py file.bin
'''
from read_raster import Raster

import sys



def best_channel(
        img_pixels, 
        channel
    ):
    '''
    This function imposes a threshold, which compares the selected channel with other 3.

    input:
    
        1. Pixelized data, of at least 3 channel, 3D array
        2. Chosen channel to filter

    output:

        >> 2D bool array (3rd dim is n channel is gone)
    '''
    channel_index = {'r': 0, 'g': 1, 'b': 2}

    chosen_index = channel_index[channel]

    other_indices = [v for v in channel_index.values() if v != chosen_index]

    _, _, n_chan = img_pixels.shape

    if n_chan > 3:

        img_pixels = img_pixels[:, :, :3]

    mask = (
        (img_pixels[:, :, chosen_index] > img_pixels[:, :, other_indices[0]]) & 

        (img_pixels[:, :, chosen_index] > img_pixels[:, :, other_indices[1]])
    )

    return mask




def plot(
        X,
        *,
        title = 'RAW',
        figsize = (15, 5)
    ):

    '''
    Applies a logic masking, returns a boolean image where:

    >> White (1): means the chosen channel has the high value at that pixel (dominates)
    >> Black (0): otherwise
    '''

    import matplotlib.pyplot as plt

    r = best_channel(X, 'r')
    g = best_channel(X, 'g')
    b = best_channel(X, 'b')

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

    #load raster and read
    raster = Raster(file_name=filename)

    X = raster.read_trim()

    plot(
        X
    )

