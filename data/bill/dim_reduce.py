
from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler


def tsne(
        X,
        params: dict
):
    
    #TSNE dimensionality reduction
    X_s = StandardScaler().fit_transform(X)

    tsne = TSNE(
        **params
    )

    X_tsne = tsne.fit_transform(X_s)

    return X_tsne



def pca(
        X,
        params: dict
):
    '''
    pca methods go heres
    '''
    
    pass









