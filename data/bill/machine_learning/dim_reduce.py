'''
Dimensionality Reduction with GPU support.
'''

import os


def tsne(
        X,
        *,
        band_list
):


    from cuml.manifold import TSNE
    import cupy as cp
    
    print(f'RUNNING ... band_list = {band_list} @ pid = {os.getpid()}.')

    X_s = X[..., [b-1 for b in band_list]]

    X_gpu = cp.asarray(X_s, dtype=cp.float32)

    tsne_gpu = TSNE(
        n_components=2,
        perplexity=60,
        learning_rate=200,
        max_iter=2000,
        init="pca",
        random_state=42,
        verbose=1
    )

    embedding = tsne_gpu.fit_transform(X_gpu)
    cp.cuda.Stream.null.synchronize()

    print(f'DONE! ... band_list = {band_list} @ pid = {os.getpid()}.')

    return cp.asnumpy( embedding )








