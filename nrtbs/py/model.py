'''
Two differnt modelling functions which function very similarly. Both take a stope index, file directory, and model type. 
NBRmodel models the NBR of a fire using all the data prior
dNBRmodel models the dNBR of the fire using all prior data
'''
from sklearn.linear_model import LinearRegression 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from dNBR import dNBR, NBR, barc_class_plot
import numpy as np
import matplotlib.pyplot as plt
import os
from misc import read_binary, extract_date, err
import matplotlib.colors
kernel = DotProduct() + WhiteKernel()

def NBRmodel(stop_index, file_dir, model_type): 
    '''
    Using a list of raster files and a model type this function models the NBR of a given fire using linear regression and 4 parameters. The function plots both an error plot of the predicted vs observed final dNBR and the predicted NBR. It also returns the fits score.
    model types:
    'lin_reg' == Linear Regression
    'KN_reg' == K Neighbor Regressor
    'gau_reg' == Gaussian Regressor
    'psl_reg' == PSL Regression
    '''
    #extracting bin files
    files = os.listdir(file_dir)
    file_list = []
    for n in range(len(files)):
        if files[n].split('.')[-1] == 'bin':
            file_list.append(files[n])
        else:
            continue
        
    sorted_file_names = sorted(file_list, key=extract_date) #sorting files by date


    vals = read_binary(f'{file_dir}/{sorted_file_names[stop_index]}') #reading each file
    width = vals[0]
    height = vals[1] 
    
    if stop_index < 0 or stop_index >= len(sorted_file_names): 
        err("bad index")  
    nbr = NBR(f'{file_dir}/{sorted_file_names[-1]}')[4]# dependent variable: compare start and end dates
    
    params = []
    for i in range(stop_index + 1):  
        params += NBR(f'{file_dir}/{sorted_file_names[i]}')[0:5] #making a list of parameters 

    X = []
    Y = []
    #print(len(params))
    #print(params[len(params)][0][0])
    for i in range(height): #making training and test data
        for j in range(width):
            x = [params[k][i][j] for k in range(len(params))]
            y = nbr[i][j]
            if not np.isnan(x).any():
                X += [x]
                Y += [y]     
            else:
                X += [[0 for k in range(len(params))]]
                Y += [0]
    if model_type == 'linear_reg': #fitting linear regression
        reg = LinearRegression().fit(X, Y)
        pred = reg.predict(X)
        data = np.zeros((height,width))
        score = reg.score(X, Y)
    elif model_type == 'KN_reg': #fitting K Neighbor Regressor
        reg = KNeighborsRegressor().fit(X,Y)
        pred = reg.predict(X)
        data = np.zeros((height,width))
        score = reg.score(X, Y)
    elif model_type == 'gau_reg': #fitting Gaussian Process Regressor
        reg = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X,Y)
        pred = reg.predict(X)
        data = np.zeros((height,width))
        score = reg.score(X, Y)
    elif model_type == 'psl_reg': #fitting PSL Regressor
        reg = PLSRegression().fit(X,Y)
        pred = reg.predict(X)
        data = np.zeros((height,width))
        score = reg.score(X, Y)     
        
    date = sorted_file_names[stop_index].split('_')[2].split('T')[0]
    for n in range(len(pred)): #going through the prediction list to plot the predicted NBR
        i = n // width
        j = n % width
        data[i][j] = pred[n]
    err = nbr - data #error 
    
    plt.figure(figsize=(15,15)) #plotting
    imratio = height/width
    
    plt.imshow(err,cmap='Greys')
    plt.colorbar(fraction=0.04525*imratio)
    plt.title(f'NBR error using stop date {date}, using {model_type}. Score: {score}')
    if not os.path.exists('NBR_model_error'):
        os.mkdir('NBR_model_error') 
    plt.tight_layout()
    plt.savefig(f'NBR_model_error/{date}_{model_type}_{sorted_file_names[stop_index]}.png')
    plt.clf()
    
    plt.imshow(data,cmap='Greys')
    plt.colorbar(fraction=0.04525*imratio)
    plt.title(f'Predicted NBR using stop date {date}, using {model_type}. Score: {score}')
    if not os.path.exists('NBR_model'):
        os.mkdir('NBR_model')
    plt.tight_layout()
    plt.savefig(f'NBR_model/{date}_{model_type}_{filenames[stop_index]}.png')
    plt.clf()
 
    
def dNBRmodel(stop_index, file_dir, model_type): 
    '''
    Using a list of raster files and a model type this function models the dNBR of a given fire using linear regression and 4 parameters. The function plots both an error plot of the predicted vs observed final dNBR and the predicted NBR. It also returns the fits score.
    model types:
    'lin_reg' == linear regression
    'KN_reg' == K Neighbor Regressor
    'gau_reg' == Gaussian Regressor
    'psl_reg' == PSL Regression
    '''
    #extracting bin files
    files = os.listdir(file_dir)
    file_list = []
    for n in range(len(files)):
        if files[n].split('.')[-1] == 'bin':
            file_list.append(files[n])
        else:
            continue
        
    sorted_file_names = sorted(file_list, key=extract_date) #sorting files by date

    vals = read_binary(f'{file_dir}/{sorted_file_names[stop_index]}') #reading each file
    width = vals[0]
    height = vals[1]  
    
    if stop_index < 0 or stop_index >= len(filenames): 
        err("bad index")  
    dnbr = dNBR(f'{file_dir}/{sorted_file_names[0]}', f'{file_dir}/{sorted_file_names[-1]}')  # dependent variable: compare start and end dates
    
    params = []
    for i in range(stop_index + 1):  
        params += NBR(f'{file_dir}/{sorted_file_names[i]}')[0:5] #making a list of parameters 
 
    X = []
    Y = [] 
    for i in range(height): #making training and test data
        for j in range(width):
            x = [params[k][i][j] for k in range(len(params))]
            y = dnbr[i][j]
            if not np.isnan(x).any():
                X += [x]
                Y += [y]     
            else:
                X += [[0 for k in range(len(params))]]
                Y += [0]

    if model_type == 'linear_reg': #fitting linear regression
        reg = LinearRegression().fit(X, Y)
        pred = reg.predict(X)
        data = np.zeros((height,width))
        score = reg.score(X, Y)    
    elif model_type == 'KN_reg': #fitting K Neighbor Regressor
        reg = KNeighborsRegressor().fit(X,Y)
        pred = reg.predict(X)
        data = np.zeros((height,width))
        score = reg.score(X, Y)

    elif model_type == 'gau_reg': #fitting Gaussian Process Regressor
        reg = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X,Y)
        pred = reg.predict(X)
        data = np.zeros((height,width))
        score = reg.score(X, Y)  
    elif model_type == 'psl_reg': #fitting PSL Regressor
        reg = PLSRegression().fit(X,Y)
        pred = reg.predict(X)
        data = np.zeros((height,width))
        score = reg.score(X, Y)      

    date = sorted_file_names[stop_index].split('_')[2].split('T')[0]
    for n in range(len(pred)): #going through the prediction list to plot the predicted dNBR
        i = n // width
        j = n % width
        data[i][j] = pred[n]
    err = dnbr - data #error
    true_barc = barc_class_plot(dnbr) 
    barc = barc_class_plot(data)
    errbarc = true_barc-barc
    
    plt.figure(figsize=(15,15)) #plotting
    imratio = height/width
    
    plt.imshow(err,cmap='Greys')
    plt.colorbar(fraction=0.04525*imratio)
    plt.title(f'dNBR error using stop date {date}, using {model_type}. Score: {score}')
    if not os.path.exists('dNBR_model_error'):
        os.mkdir('dNBR_model_error') 
    plt.tight_layout()
    plt.savefig(f'dNBR_model_error/{date}_{model_type}_{sorted_file_names[stop_index]}.png')
    plt.clf()
    
    plt.imshow(data,cmap='Greys')
    plt.colorbar(fraction=0.04525*imratio)
    plt.title(f'Predicted dNBR using stop date {date}, using {model_type}. Score: {score}')
    if not os.path.exists('dNBR_model'):
        os.mkdir('dNBR_model')
    plt.tight_layout()
    plt.savefig(f'dNBR_model/{date}_{model_type}_{sorted_file_names[stop_index]}.png')
    plt.clf()
    
    cmap = matplotlib.colors.ListedColormap(['green','yellow','orange','red']) 
    plt.imshow(barc,vmin=0,vmax=3, cmap=cmap)
    plt.title(f'BARC 256 burn severity model using stop date {date}, using {model_type}. Score: {score}')
    if not os.path.exists('BARC_model'):
        os.mkdir('BARC_model')
    plt.tight_layout()
    plt.savefig(f'BARC_model/{date}_{model_type}_{sorted_file_names[stop_index]}.png')
    plt.clf()
    
    plt.imshow(errbarc, vmin=0,vmax=3,cmap=cmap)
    plt.title(f'BARC 256 burn severity error using stop date {date}, using {model_type}. Score: {score}')
    if not os.path.exists('BARC_model_error'):
        os.mkdir('BARC_model_error')
    plt.tight_layout()
    plt.savefig(f'BARC_model_error/{date}_{model_type}_{sorted_file_names[stop_index]}.png')
    plt.clf()
