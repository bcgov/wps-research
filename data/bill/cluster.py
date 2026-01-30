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
        fitted_cluster
):
    '''
    fitted cluster are fitted from hdbscan fit
    '''

    from cuml.cluster.hdbscan import approximate_predict
    
    new_labels, strengths = approximate_predict(fitted_cluster, X)

    return new_labels, strengths