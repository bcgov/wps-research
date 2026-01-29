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
    
    # from cuml.manifold import TSNE
    
    # print(f'running: {band_list}: pid = {os.getpid()}.')

    # X_s = StandardScaler().fit_transform(
    #     X[..., [b-1 for b in band_list]]
    # )

    # tsne = TSNE(
    #     n_components=2,
    #     perplexity=30,
    #     learning_rate=200,
    #     n_iter=1000,
    #     verbose=1
    # )

    # embedding = tsne.fit_transform(X_s)

    # print(f'DONE: {band_list}: pid = {os.getpid()}.')

    # return embedding


    from cuml.manifold import TSNE
    import cupy as cp
    
    print(f'running: {band_list}: pid = {os.getpid()}.')

    X_s = StandardScaler().fit_transform(
        X[..., [b-1 for b in band_list]]
    )

    X_gpu = cp.asarray(X_s)

    tsne_gpu = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1000,
        init="random",
        verbose=1
    )

    embedding = tsne_gpu.fit_transform(X_gpu)
    cp.cuda.Stream.null.synchronize()

    print(f'DONE: {band_list}: pid = {os.getpid()}.')

    return cp.asnumpy( embedding )



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








