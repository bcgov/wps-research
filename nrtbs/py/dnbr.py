'''
calculates burn severity using the dNBR of a frame. First two functions are used to extract data from a bin file and return it. dNBR calculates the dNBR of the provided frames and removes water and reduces noise.
Class_plot plots the burn severity of the given dNBR using BARC thresholds.
time_series plots the time series of BARC plots for the provided file directory and start frame date
'''

from misc import err, read_binary, extract_date
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import os
from data_to_raster import write_matrix_to_tif

def nbr_full(file_name):
    """
    Takes a binary file and returnes all bands as well as NBR,NBRSWIR and NVDI
    """
    vals = read_binary(file_name) 
    data = vals[3]
    width = vals[0]
    height = vals[1]
    band_names = ['B1','B2','B3','B4','B5','B6','B7','B8A','B08','B09','B11','B12']
    for b in band_names:
        exec(f"{b} = np.zeros(({height}, {width}))", globals())
    
    # extracts band values into 2D arrays
    for i in range(height):
        for j in range(width):
            B12[i][j] = data[width*height*11 + width*i+j]
            B11[i][j] = data[width*height*10 + width*i+j]
            B09[i][j] = data[width*height*9 + width*i+j]
            B08[i][j] = data[width*height*8 + width*i+j]
            B8A[i][j] = data[width*height*7 + width*i+j]
            B7[i][j] = data[width*height*6 + width*i+j]
            B6[i][j] = data[width*height*5 + width*i+j]
            B5[i][j] = data[width*height*4 + width*i+j]
            B4[i][j] = data[width*height*3 + width*i+j]
            B3[i][j] = data[width*height*2 + width*i+j]
            B2[i][j] = data[width*height*1 + width*i+j]
            B1[i][j] = data[width*height*0 + width*i+j]

    NBR = (B08-B12)/(B08+B12)#calculating NBR
    NDVI = (B08 - B4)/(B08 + B4)#calculating NDVI
    
    nbrswir = (B11-B12-0.02)/(B11+B12+0.1)

    return [B1,B2,B3,B4,B5,B6,B7,B8A,B08,B09,B11,B12,NBR,NDVI,nbrswir, height,width]
            

def NBR(file_name, harmonized=True):
    '''
    Takes binary file and returns the band values as well as the NBR.
    >>> NBR('S2A_MSIL1C_20210902T190911_N0301_R056_T10UFB_20210902T225534.bin')
    '''
    width, height, _, data = read_binary(file_name)
    cube = np.asarray(data, dtype=np.float32).reshape(4, height, width)
    B12, B11, B09, B08 = cube[0], cube[1], cube[2], cube[3]

    if harmonized:
        B12 = B12 - 1000
        B11 = B11 - 1000
        B09 = B09 - 1000
        B08 = B08 - 1000

    with np.errstate(divide='ignore', invalid='ignore'):
        NBR = (B08 - B12) / (B08 + B12)
        nbrswir = (B11 - B12 - 0.02) / (B11 + B12 + 0.1)

    return [B12, B11, B09, B08, NBR, height, width, nbrswir]
            
            
def dNBR(start_frame, end_frame):
    '''
    Takes the start and end binary files and returns the dNRB.
    >>> dNBR('S2B_MSIL1C_20210626T185919_N0300_R013_T10UFB_20210626T211041.bin', 'S2A_MSIL1C_20210907T190911_N0301_R056_T10UFB_20210902T225534.bin')
    '''
    if type(start_frame) == str: # determining the type of the data given
        predata = NBR(start_frame)
    else:
        predata = start_frame

    postdata = NBR(end_frame)
    preNBR = predata[4]
    postNBR = postdata[4]
    preswir = predata[7]
    postswir = postdata[7]
    dNBR = preNBR - postNBR #calculating dNBR
    dNBRSWIR = preswir - postswir # calculating dNBRSWIR
    
    #removing water and some noise
    # for i in range(len(dNBR)):
    #     for j in range(len(dNBR[0])):
    #         if predata[0][i][j] <= 100 or dNBRSWIR[i][j] < 0.1:
    #             dNBR[i][j] = 0
    #         else:
    #             continue

    return dNBR

def barc_class_plot(dNBR, start_date, end_date, title='Not given'): 
    '''
    Plots the BARC 256 burn severity of the provided dNBR and saves it as a png
    '''
    
    scaled_dNBR = (dNBR*1000+275)/5 #scalling dNBR
    class_plot = np.zeros((len(scaled_dNBR),len(scaled_dNBR[0])))
    un_tot = 0
    low_tot = 0
    med_tot = 0
    high_tot = 0
    for i in range(len(scaled_dNBR)): #making classifications
        for j in range(len(scaled_dNBR[0])):
            if scaled_dNBR[i][j] < 76:
                class_plot[i][j] = 1
                un_tot += 1
            elif 76 <= scaled_dNBR[i][j] < 110:
                class_plot[i][j] = 2
                low_tot += 1
            elif 110 <= scaled_dNBR[i][j] < 187:
                class_plot[i][j] = 3
                med_tot += 1
            elif np.isnan(scaled_dNBR[i][j]):
                class_plot[i][j] = float('nan')
            else:
                class_plot[i][j] = 4
                high_tot += 1
    
    #calculating percentages           
    tot = un_tot+low_tot+med_tot+high_tot
    un_per = round(100*un_tot/tot,1)
    low_per = round(100*low_tot/tot,1)
    med_per = round(100*med_tot/tot,1)
    high_per = round(100*high_tot/tot,1)
    
    if not os.path.exists(title):
        os.mkdir(title)
    #plotting
    cmap = matplotlib.colors.ListedColormap(['green','yellow','orange','red'])
    plt.figure(figsize=(15,15))
    plt.imshow(class_plot,vmin=1,vmax=4,cmap=cmap)
    plt.title(f'BARC 256 burn severity, start date:{start_date}, end date:{end_date}')
    plt.scatter(np.nan,np.nan,marker='s',s=100,label=f'Unburned {un_per}%',color='green')
    plt.scatter(np.nan,np.nan,marker='s',s=100,label=f'Low {low_per}%' ,color='yellow')
    plt.scatter(np.nan,np.nan,marker='s',s=100,label=f'Medium {med_per}%',color='orange')
    plt.scatter(np.nan,np.nan,marker='s',s=100,label=f'High {high_per}%',color='red')
    plt.legend(fontsize="20")
    plt.tight_layout()
    plt.savefig(f'{title}/{end_date}_BARC_classification.png')
    plt.close()
    print('+w', f'{title}/{end_date}_BARC_classification.png')
    return class_plot


def barc_time_series(directory,start_date,title='BARC'):
    '''
    Takes a Directory and plots a time serise of BARC plots with the provided start date 
    '''
    #sorting files
    files = os.listdir(directory)
    file_list = []
    for n in range(len(files)):
        if files[n].split('.')[-1] == 'bin':
            file_list.append(files[n])
        else:
            continue

    sorted_file_names = sorted(file_list, key=extract_date)
    
    #finding start date index
    for i in range(len(sorted_file_names)):
        if extract_date(sorted_file_names[i]) == str(start_date):
            index = i
            break
        else:
            index = None
            continue
    if index == None:
        err('Invalid start date')
    
    #calculating start frame nbr
    start_file = sorted_file_names[index]
    start_frame = NBR(f'{directory}/{start_file}')
    
    #making BARC plots
    i = 0
    for file in sorted_file_names[index +1:]:
        print("time_series,file=",file) 
        i += 1  
        dnbr = dNBR(start_frame, f'{directory}/{file}')
        end_date = extract_date(file)

        data = barc_class_plot(dnbr,start_date,end_date,title)  

        if True: # i == len(sorted_file_names[index +1:]):
            # print('Writing data to Tiff')
            
            barc_folder_name = '_'.join(title.split('_')[:-1]) + '_barcs'
            if not os.path.exists(barc_folder_name):
                os.mkdir(barc_folder_name)

            if not os.path.exists(title+'_barcs'):
                os.mkdir(title+'_barcs')

            write_matrix_to_tif(data, f'{directory}/{file}', f'{title}_barcs/BARC_{title}_{start_date}_{end_date}_BARC.tif') 

def barc_to_tiff(fire_dir, start_date, end_date):
    title = f'{fire_dir.strip("_cut")}_barcs'
    files = os.listdir(fire_dir)
    file_list = []
    for n in range(len(files)):
        if files[n].split('.')[-1] == 'bin':
            file_list.append((int(extract_date(files[n])), f'{fire_dir}/{files[n]}'))
        else:
            continue
    
    #sorted_file_names = sorted(file_list, key=extract_date) #sorting
    
    for file in file_list:
        if file[0] == start_date:
            start_file = file[1]
        elif file[0] == end_date:
            end_file = file[1]
    
    dnbr = dNBR(start_file, end_file)
    data = class_plot(dnbr, start_date, end_date, title)
    write_matrix_to_tif(data, end_file, f'{title}/{end_date}_barc.tif')
