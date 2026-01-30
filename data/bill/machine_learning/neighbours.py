'''
Neighbours algorithms with GPU support.
'''

def knn_regressor(
        X, 
        y,
        **params
):
    '''
    Description
    -----------
    Regress based on nearest neighbours.
    '''

    from cuml.neighbors import KNeighborsRegressor

    reg = KNeighborsRegressor(
        **params
    )

    reg.fit(X, y)

    return reg

