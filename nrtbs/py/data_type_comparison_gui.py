'''
Interactive GUI which allows you to compare ban values as a time series for multiple data sets. Take a list of directories, each one containing a time series of bin files. When the plot is clicked a box will apear on the plot and the average value for each band in the square is plotted as a time series for each data set.
>>> interactive_time_serise(['bins1','bins2','bins3'], 'image',10)
'''

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dNBR import NBR
from operator import add, sub
import datetime
import numpy as np
import math
from misc import extract_date
import os

# plot parameters
fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2, figsize=(15,8))
clicks = []
plot_colors = ['b','r','y','k','c','m']

def interactive_time_serise(file_dir_list,plot_type:str('image or nbr'), width):
    '''
    Takes a list of directories and plots an interactive image of the nbr or 3-band color image. Clicking the plot will create a plot of the time serise for the B12, B11, B09, B08, and NBR bands for each of the directories as an average inside the produced square which has a width specified when calling the function. Clicking again produces another time serise for the new square.
    '''
    global file_dirs #defining global variables
    file_dirs = file_dir_list
    
    global file_arr
    file_arr = []
    for file_dir in file_dir_list: #creating an array of file names
        files = os.listdir(file_dir)
        file_list = []
        for n in range(len(files)):
            if files[n].split('.')[-1] == 'bin':
                file_list.append(files[n])
            else:
                continue
        sorted_file_names = sorted(file_list, key=extract_date) #sorting files by date
        file_arr.append(sorted_file_names)

    global param_arr
    param_arr = []
    for n in range(len(file_arr)):
        params = []
        for file in file_arr[n]:
            params.append(NBR(f'{file_dir_list[n]}/{file}'))
        param_arr.append(params)
        

    global square_width #global square width
    square_width = width
    
    global plot
    plot = plot_type
    
    global data
    
    data = NBR(f'{file_dir_list[0]}/{file_arr[0][-1]}')
    
    if plot == 'image': #plotting image or nbr 
        image = np.stack([scale(data[0]),scale(data[1]),scale(data[2])], axis=2)
        ax1.imshow(image)

    elif plot == 'nbr':
        ax1.imshow(data[4], cmap='grey')
        
    
    cid = fig.canvas.mpl_connect('button_press_event', on_click) #calling gui
    
    plt.show()
    
def param_plots(clicks, width):
    '''
    takes the list of click locations and square_width and plots the B12, B11, B09, B08, and NBR of the mean value timeserise in a box with side lenght = square_width
    '''
    ax = [ax2,ax3,ax4,ax5,ax6]
    band_names = ['B12', 'B11', 'B09', 'B08', 'NBR']
    
    y = int(clicks[-1][1]) #click coordinates
    x = int(clicks[-1][0])
    
    ax[0].cla() #clearing plots
    ax[1].cla()
    ax[2].cla()
    ax[3].cla()
    ax[4].cla()
    for m in range(len(param_arr)): #calculating mean for each directory
        b = [[] for i in range(len(band_names))]
        mean = [[] for i in range(len(band_names))]
        std = [[] for i in range(len(band_names))]
        time = []
        for file in range(len(param_arr[m])):#loops finding pixel values for each parameter in the square
            for band in range(len(band_names)):
                b[band] = []
        
            date  = datetime.datetime.strptime(file_arr[m][file].split('_')[2].split('T')[0],'%Y%m%d')
            for i in range(y,y+width):
                for j in range(x,x+width):
                    for n in range(len(band_names)):
                        b[n] += [param_arr[m][file][n][i][j]]
                    
            for band in range(len(band_names)):
                mean[band] += [np.nanmean(b[band])]
                std[band] += [np.nanstd(b[band])]        
            time += [date]
    
        for band in range(len(band_names)): #plotting each time serise
            ax[band].plot(time, mean[band], color=f'{plot_colors[m]}',label=f'{file_dirs[m]} mean at ({x},{y})')
            ax[band].plot(time, list(map(add,mean[band], std[band])), color=f'{plot_colors[m]}', linestyle='dashed')
            ax[band].plot(time, list(map(sub,mean[band], std[band])), color=f'{plot_colors[m]}', linestyle='dotted')

            ax[band].legend()
            ax[band].set_title(band_names[band])

    plt.tight_layout()
    plt.show()

def on_click(event):
    print(f"Clicked at: {event.xdata}, {event.ydata}")
    if event.inaxes is not None:  # Check if the click is inside the plot area
        # Store the click coordinates
        clicks.append((event.xdata, event.ydata))
        print(f"Clicked at: {event.xdata}, {event.ydata}")
        
        # Create a square patch
        square = patches.Rectangle((event.xdata, event.ydata), square_width, square_width, 
                                   linewidth=1, edgecolor='r', facecolor='none')
        ax1.cla()
        if plot == 'image':
            image = np.stack([scale(data[0]),scale(data[1]),scale(data[2])], axis=2)
            ax1.imshow(image)

        elif plot == 'nbr':
            ax1.imshow(data[4], cmap='grey')
        
        else:
            print('invalid image type')
    
        ax1.add_patch(square)  # Add the square to the plot
        fig.canvas.draw()  # Update the plot
        param_plots(clicks,square_width)
        
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


