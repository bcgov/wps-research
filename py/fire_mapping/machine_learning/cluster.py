'''
Clustering with GPU support.
'''

def hdbscan_fit(
        X,
        **params
):

    from cuml.cluster.hdbscan import HDBSCAN

    cluster = HDBSCAN(**params)

    fitted_pred = cluster.fit_predict(X)

    cluster.generate_prediction_data()

    return cluster, fitted_pred



def hdbscan_approximate(
        X,
        fitted_cluster,
        chunk_size=500_000
):
    '''
    fitted clusters are fitted from hdbscan fit.
    Processes in chunks to avoid GPU out-of-memory on large images.
    '''
    import numpy as np
    from cuml.cluster.hdbscan import approximate_predict

    n = X.shape[0]
    if n <= chunk_size:
        new_labels, strengths = approximate_predict(fitted_cluster, X)
        return new_labels, strengths

    all_labels    = np.empty(n, dtype=np.int32)
    all_strengths = np.empty(n, dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        labels_chunk, strengths_chunk = approximate_predict(
            fitted_cluster, X[start:end])
        all_labels[start:end]    = labels_chunk
        all_strengths[start:end] = strengths_chunk

    return all_labels, all_strengths