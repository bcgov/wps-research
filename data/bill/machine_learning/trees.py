'''
Tree-based methods with GPU support.
'''

def rf_regressor(
        X, 
        y,
        **params
):
    '''
    Description
    -----------
    Regress based using random forest.
    '''

    from cuml.ensemble import RandomForestRegressor

    reg = RandomForestRegressor(
        **params
    )

    reg.fit(X, y)

    return reg