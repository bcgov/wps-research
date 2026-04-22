'''
Creates TSNE plots for the provided data (not complete)
'''
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from  misc import read_binary

def data(start_file,end_file):
    '''
    Creates a 4d data list which can be used by scikit learns tSNE package. Also classifies each data point with the BARC threshold values
    '''

    prevals = read_binary(start_file) #reading files
    predata = prevals[3]
    postvals = read_binary(end_file)
    postdata = postvals[3] 
    width = prevals[0]
    height = prevals[1]
    prebands = []
    postbands = []
    data = []
    USE_NTH = 10
    label = []
    colors = []
    for pixle in range(width*height):
        if pixle % USE_NTH == 0: #only selecting every NTH point
            prebands = []
            postbands = []
            for band in range(4):
                prebands.append(predata[pixle*band])
                postbands.append(postdata[pixle*band])
            data.append(prebands) #adding to data
            dnbr = (prebands[0]-prebands[3])/(prebands[0]+prebands[3]) - (postbands[0]-postbands[3])/(postbands[0]+postbands[3])
            scaled_dNBR = (dnbr*2000+275)/5
            if scaled_dNBR < 76: #calculating BARC
                label.append('unburned')
                colors.append('green')
            elif 76 <= scaled_dNBR < 110:
                label.append('low')
                colors.append('yellow')
            elif 110 <= scaled_dNBR < 187:
                label.append('med')
                colors.append('orange')
            else:
                label.append('high')
                colors.append('red')
        else:
            continue
    return [np.array(data),label,colors]

def create_tsne_embedding(data, n_components=2):
    """
    Create a t-SNE embedding from the given data.

    Parameters:
    data (np.ndarray): The input data array of shape (n_samples, n_features).
    n_components (int): The dimension of the embedded space (default is 2).
    perplexity (float): The perplexity parameter for t-SNE (default is 30).
    learning_rate (float): The learning rate for t-SNE (default is 200).
    n_iter (int): The number of iterations for optimization (default is 1000).

    Returns:
    np.ndarray: The embedded data array of shape (n_samples, n_components).
    """
    tsne = TSNE(n_components=n_components)
    tsne_results = tsne.fit_transform(data)
    return tsne_results

def plot_tsne_embedding(embedded_data, labels, colors):
    """
    Plot the t-SNE embedding.

    Parameters:
    embedded_data (np.ndarray): The embedded data array of shape (n_samples, n_components).
    labels (np.ndarray): The labels for each data point (default is None).
    """
    plt.figure(figsize=(15, 15))
    plt.scatter(embedded_data[:, 0],embedded_data[:, 1],c=colors,s=50)
    plt.scatter(np.nan,np.nan,color='green', label='Unburned')
    plt.scatter(np.nan,np.nan,color='yellow', label='Low')
    plt.scatter(np.nan,np.nan,color='orange', label='Medium')
    plt.scatter(np.nan,np.nan,color='red', label='High')
    plt.title('BARC tSNE')
    plt.legend(fontsize="20")
    plt.show()


data = data('L2/small/S2B_MSIL2A_20210626T185919_N0300_R013_T10UFB_20210626T211041.bin','L2/small/S2B_MSIL2A_20210907T190929_N0301_R056_T10UFB_20210907T224046.bin')
data_4d = data[0]
label = data[1]
colors = data[2]

tsne_2d = create_tsne_embedding(data_4d)

# Plot the t-SNE embedding
plot_tsne_embedding(tsne_2d,label,colors)

