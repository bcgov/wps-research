from misc import exist, read_hdr, read_float, hdr_fn, read_binary
import numpy as np
import matplotlib.pyplot as plt
import math
import os


def scale(X):
    # default: scale a band to [0, 1]  and then clip
    mymin = np.nanmin(X) # np.nanmin(X))
    mymax = np.nanmax(X) # np.nanmax(X))
    X = (X-mymin) / (mymax - mymin)  # perform the linear transformation

    X[X < 0.] = 0.  # clip
    X[X > 1.] = 1.

    # use histogram trimming / turn it off to see what this step does!
    if  True:
        values = X.ravel().tolist()
        values.sort()
        n_pct = 1. # percent for stretch value
        frac = n_pct / 100.
        lower = int(math.floor(float(len(values))*frac))
        upper = int(math.floor(float(len(values))*(1. - frac)))
        mymin, mymax = values[lower], values[upper]
        X = (X-mymin) / (mymax - mymin)  # perform the linear transformation
    
    return X
    

def plot(file):
    '''
    Takes a list of binary raster files organized by date and plots an image using the B12, B11, and B09 bands. Also plots the NBR of each frame as well as the dNBR of each frame except the first (first frame would have dNBR=0). Function places each files into three directories: 'images', 'NBR', and 'dNBR'.
    >>> plot(['S2B_MSIL1C_20210626T185919_N0300_R013_T10UFB_20210626T211041.bin',...,'S2B_MSIL1C_20210907T190929_N0301_R056_T10UFB_20210907T224046.bin'])
    '''
    band_list = ['C11','C12_imag', 'C12_real', 'C22']
    for file in file_list:
        for band in band_list:
            vals = read_binary(f'2018_shovel/{file}/{band}.bin') #reading each file
            data = vals[3]
            #print(data)
            width = vals[0]
            height = vals[1]
            bands = vals[2]
            C = np.zeros((height,width))
            for i in range(height):
                for j in range(width):
                    C[i][j] = data[width*i+j]
            if band == 'C12_imag':
                C12_im = C
            elif band == 'C12_real':
                C12_re = C
            elif band == 'C11':
                C11 = C
            elif band == 'C22':
                C22 = C                
            ''' 
            plt.figure(figsize=(15,15))
            plt.imshow(C, cmap='grey') #Plotting the image
            plt.title(f'Mystery date: {file}, band: {band}')
            imratio=height/width
            plt.colorbar(fraction=0.046*imratio)
            if not os.path.exists(f'{file}'):
                os.mkdir(f'{file}')
            plt.tight_layout()
            plt.savefig(f'{file}/{band}.png')
            plt.clf() 
            '''
        data = np.sqrt(C12_im**2 + C12_re**2)
        x = scale(C11)
        y = scale(data)
        z = scale(C22)
        
        plt.figure(figsize=(15,15)) #setting figure parameters
        
        image = np.stack([x,y,z], axis=2)
        plt.imshow(data) #Plotting the image
        plt.title(f'Mystery date: {file},')
        imratio=height/width
        #plt.colorbar(fraction=0.046*imratio)
        if not os.path.exists(f'{file}'):
            os.mkdir(f'{file}')
        plt.tight_layout()
        plt.savefig(f'{file}/radio.png')
        plt.clf()

file_list = ['20170720','20190718','20160721','20161208','20180830','20181206','20190829']
plot(file_list)
