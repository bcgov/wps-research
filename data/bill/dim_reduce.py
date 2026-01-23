'''
Dimensionality reduction and Parallel Execution.
'''

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import os

def tsne(
        X,
        *,
        band_list,
        params: dict
):
    '''
    Just a TSNE implementation.
    '''
    from sklearn.manifold import TSNE

    print(f'{band_list}: pid = {os.getpid()} running...') #Write wrapper?
    
    #TSNE dimensionality reduction
    X_s = StandardScaler().fit_transform(X)

    tsne = TSNE(
        **params
    )

    X_tsne = tsne.fit_transform(X_s)

    print(f'{band_list}: pid = {os.getpid()} ...Done.')

    return X_tsne



def pca(
        X,
        *,
        band_list,
        params: dict
):
    '''
    Just a PCA implementation.
    '''
    from sklearn.decomposition import PCA
    
    print(f'{band_list}: pid = {os.getpid()} running...')

    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('pca', PCA(**params))
    ])

    X_pca = pipe.fit_transform(X)

    print(f'{band_list}: pid = {os.getpid()} ...Done.')

    return pipe, X_pca



def parDimRed(
        tasks: list[tuple]
):
    '''
    Description
    -----------
    Performs parallel execution to speed up independent executions.
    
    Designed specifically for Sentinel-2 band data dimensionality reduction.


    Parameters
    ----------
    X_dict: dictionary of data, each will be assigned to workers.

    method: a method to receives.
    '''

    from concurrent.futures import ProcessPoolExecutor, as_completed

    from time import time


    data_size = len(tasks)

    t0 = time()

    futures = {}

    with ProcessPoolExecutor(max_workers=4) as pool:

        for band_list, X, method, params in tasks:
            f = pool.submit(

                method,

                X,
                band_list=band_list,
                params=params

            )
            futures[f] = band_list


        embedding_dict = {}
        model_dict = {}

        for f in as_completed(futures):

            model, embedding = f.result()

            band_lst = futures[f]

            embedding_dict[str(band_lst)] = embedding
            model_dict[str(band_lst)] = model

    print(f'{method} on {data_size} data took {time() - t0:.2f} s')

    return embedding_dict, model_dict








