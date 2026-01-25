'''
Dimensionality reduction and Parallel Execution.
'''

import os
from sklearn.preprocessing import StandardScaler


def tsne(
        X,
        *,
        band_list
):
    
    from openTSNE import affinity, TSNEEmbedding
    import numpy as np
    
    print(f'running: {band_list}: pid = {os.getpid()}.')

    X_s = StandardScaler().fit_transform(X)

    aff = affinity.PerplexityBasedNN(
        X_s,
        perplexity = 60,
        metric = "euclidean",
        n_jobs = 4,
        random_state = 123
    )

    init = TSNEEmbedding(
        np.random.normal(0, 1e-4, (X.shape[0], 2)),
        aff
    )

    embedding = init.optimize(
        n_iter=1000,
        exaggeration=10,
        learning_rate=200
    )

    print(f'DONE: {band_list}: pid = {os.getpid()}.')

    return embedding



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

    with ProcessPoolExecutor(max_workers=12) as pool:

        for band_list, X in tasks:

            f = pool.submit(
                tsne,
                X,
                band_list=band_list
            )

            futures[f] = band_list


        embed_dict = {}

        for f in as_completed(futures):

            embedding = f.result()

            band_lst = futures[f]

            embed_dict[str(band_lst)] = embedding

    print(f'OpenTSNE on {data_size} band combinations took {time() - t0:.2f} s')

    return embed_dict








