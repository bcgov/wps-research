'''
Several functions which help with creating accurate burn masks for fires suing start and end frames
'''
from misc import read_binary, extract_date
import numpy as np
import matplotlib.pyplot as plt
import os
from dNBR import dNBR, NBR
from operator import add, sub
import datetime

def burnmask(start_file,end_file,threshold):
    '''
    creates a burn mask of the provided fire using the fire start and end frames and provided dNBR threshold
    >>>burnmask('raster_data/small/S2B_MSIL1C_20210626T185919_N0300_R013_T10UFB_20210626T211041.bin', 'raster_data/small/S2B_MSIL1C_20210907T190929_N0301_R056_T10UFB_20210907T224046.bin')
    '''
    vals = read_binary(start_file)
    dnbr = dNBR(start_file,end_file)
    mask = dnbr >= threshold

    return mask

def param_plots(file_dir,threshold):
    '''
    Makes plots of the burned and unburned mean and standard deveation of the 4 parameters calculated using the NBR function, B12, B11, B09, B08, and NBR.
    '''
    band_names = ['B12', 'B11', 'B09', 'B08', 'NBR']
    burned_b = [[] for i in range(len(band_names))]
    unburned_b = [[] for i in range(len(band_names))]
    burned_mean = [[] for i in range(len(band_names))]
    unburned_mean = [[] for i in range(len(band_names))]
    burned_std = [[] for i in range(len(band_names))]
    unburned_std = [[] for i in range(len(band_names))]
    
    #extracting bin files
    files = os.listdir(file_dir)
    file_list = []
    for n in range(len(files)):
        if files[n].split('.')[-1] == 'bin':
            file_list.append(files[n])
        else:
            continue
        
    sorted_file_names = sorted(file_list, key=extract_date) #sorting files by date

    
    mask = burnmask(f'{file_dir}/{sorted_file_names[0]}',f'{file_dir}/{sorted_file_names[-1]}',threshold) #calculating the mask ie. where is burned/unburned
    
    time = []
    
    
    for file in sorted_file_names:#loops finding pixel values for each parameter and sorting them to burned or unburned
        for band in range(len(band_names)):
            burned_b[band] = []
            unburned_b[band] = []
        
        date  = datetime.datetime.strptime(file.split('_')[2].split('T')[0],'%Y%m%d')
        params = NBR(f'{file_dir}/{file}')
        for i in range(len(mask)):
            for j in range(len(mask[0])):
                for band in range(len(band_names)):
                    if mask[i][j]:
                        burned_b[band] += [params[band][i][j]]
                    else:
                        unburned_b[band] += [params[band][i][j]]
                        
        for band in range(5):
            burned_mean[band] += [np.nanmean(burned_b[band])]
            burned_std[band] += [np.nanstd(burned_b[band])]
            
            unburned_mean[band] += [np.nanmean(unburned_b[band])]
            unburned_std[band] += [np.nanstd(unburned_b[band])]   
            
        time += [date]

    plt.figure(figsize=(15,15)) #plotting
    
    band_title = ['B12', 'B11', 'B09', 'B08', 'NBR']
    for band in range(len(band_names)):
        plt.plot(time, burned_mean[band], color='red',label='Burned mean')
        plt.plot(time, list(map(add,burned_mean[band], burned_std[band])), color='red', linestyle='dashed', label='Burned mean +')
        plt.plot(time, list(map(sub,burned_mean[band], burned_std[band])), color='red', linestyle='dotted', label='Burned mean -')
        
        plt.plot(time, unburned_mean[band], color='green',label='Unburned mean')
        plt.plot(time, list(map(add, unburned_mean[band], unburned_std[band])), color='green', linestyle='dashed', label='Unburned mean +')
        plt.plot(time, list(map(sub,unburned_mean[band], unburned_std[band])), color='green', linestyle='dotted', label='Unburned mean -')            
        plt.legend()
        plt.title(f'{band_title[band]} burned and unburned')
        plt.tight_layout()
        plt.savefig(f'{band_title[band]}_timeserise.png')
        plt.clf()
        
def burn_unburn(file_dir):
    #extracting bin files
    files = os.listdir(file_dir)
    file_list = []
    for n in range(len(files)):
        if files[n].split('.')[-1] == 'bin':
            file_list.append(files[n])
        else:
            continue
        
    sorted_file_names = sorted(file_list, key=extract_date) #sorting files by date
    
    nbrs = []
    b11s = []
    b12s = []
    b8s = []
    nbrswirs = []
    for file in sorted_file_names:
        data = NBR(f'{file_dir}/{file}')
        nbrs.append(data[4])
        b11s.append(data[1])
        b12s.append(data[0])
        b8s.append(data[3])
        nbrswirs.append((data[1]-data[0]-0.02)/(data[1]+data[0]+0.1))
    
    height = len(nbrs[0])
    width = len(nbrs[0][0])
    burned = np.zeros((height,width))
    for frame in range(1,len(nbrs)):
        date  = sorted_file_names[frame].split('_')[2].split('T')[0]
        for i in range(height-5):
            for j in range(width-5):
                if (nbrs[frame-1][i][j] - nbrs[frame][i][j]) > 0.105 and (nbrswirs[frame-1][i][j] - nbrswirs[frame][i][j]) > 0.1 and b12s[frame][i][j] > 100 and b8s[frame][i][j] < 2000:
                    burned[i][j] = True
        plt.imshow(burned,cmap='grey') 
        plt.savefig(f'{date}.png')
        
