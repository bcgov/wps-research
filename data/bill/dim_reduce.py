'''
Dimensionality reduction and Parallel Execution.
'''

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

import os

def tsne(
        X,
        *,
        info,
        params: dict
):
    '''
    Just a TSNE implementation.
    '''

    print(f'{info}: pid = {os.getpid()} running...') #Write wrapper?
    
    #TSNE dimensionality reduction
    X_s = StandardScaler().fit_transform(X)

    tsne = TSNE(
        **params
    )

    X_tsne = tsne.fit_transform(X_s)

    print(f'{info}: pid = {os.getpid()} ...Done.')

    return X_tsne



def pca(
        X,
        *,
        info,
        params: dict
):
    '''
    Just a PCA implementation.
    '''
    
    pass



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

    #Determine number of workers
    data_size = len(tasks)

    # n_workers = min(data_size, os.cpu_count() // 2)

    t0 = time()

    futures = {}

    with ProcessPoolExecutor(max_workers=4) as pool:

        for info, X, method, params in tasks:
            f = pool.submit(
                method,
                X,
                info=info,
                params=params
            )
            futures[f] = info

        results = {}
        for f in as_completed(futures):
            info = futures[f]
            results[str(info)] = f.result()

    print(f'{method} on {data_size} data took {time() - t0:.2f} s')


    return results








