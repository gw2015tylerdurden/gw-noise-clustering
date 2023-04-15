import numpy as np
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import pdist, squareform

def classifnp(data, clustering, method='centroid', cdist=None, centroids=None, nnk=1):
    data = np.array(data)
    clustering = np.array(clustering)
    k = np.max(clustering) + 1  # max clustering label
    n, p = data.shape
    # unclustered data indices
    topredict = clustering < 0
    predicted_clustering = pd.Series(clustering)

    ##### Average linkage
    if method == 'averagedist':
        if cdist is None:
            cdist = squareform(pdist(data))
        else:
            cdist = np.array(cdist)

        # [unclustered_data_distance, class]
        prmatrix = np.zeros((sum(topredict), k))
        for j in range(k):
            unclustered_pred_j = cdist[topredict][:, clustering == j]
            prmatrix[:, j] = np.mean(unclustered_pred_j, axis=1)

        clpred = np.nanargmin(prmatrix, axis=1)
        predicted_clustering[topredict] = clpred

    #### Kmeans, PAM, specc, ...

    return predicted_clustering


def nselectboot(data, B=50, distances=None, clustermethod="kmeans",
                classification="averagedist", centroidname=None,
                krange=range(2, 11), count=False,
                nnk=1, largeisgood=False, **kargs):

    if clustermethod == 'kmeans':
        model = KMeans
    elif clustermethod == 'specc':
        model = SpectralClustering
    else:
        raise ValueError('Invalid clustermethod')

    data = pd.DataFrame(data)
    stab = np.zeros((B, max(krange)))
    n = data.shape[0]

    for k in krange:
        if count:
            print(f"{k} clusters")

        for b in range(B):
            if count:
                print(b)

            # test
            d1 = np.repeat(np.arange(0, n+1, 2), 2)[:n]
            d2 = np.repeat(np.arange(1, n+1, 2), 2)[:n]
            d2[-1] = n - 1

            #d1 = np.random.choice(n, n, replace=True)
            #d2 = np.random.choice(n, n, replace=True)

            dmat1 = data.iloc[d1, :]
            dmat2 = data.iloc[d2, :]

            #clm1 = model(n_clusters=k, **kargs).fit(dmat1)
            #clm2 = model(n_clusters=k, **kargs).fit(dmat2)

            cj1 = pd.Series(-1, index=range(n))
            cj2 = pd.Series(-1, index=range(n))

            #cj1[d1] = clm1.labels_
            #cj2[d2] = clm2.labels_
            # test

            cj1[d1] = np.tile(np.arange(0, 22), int(n / 21))[:n]
            cj2[d2] = np.tile(np.arange(0, 22), int(n / 21))[:n]

            cj1 = classifnp(data, cj1, method=classification, centroids=None, nnk=nnk)
            cj2 = classifnp(data, cj2, method=classification, centroids=None, nnk=nnk)

            ctable = np.array(pd.crosstab(cj1, cj2))
            nck1 = ctable.sum(axis=1)
            stab[b, k - 1] = np.sum(nck1 ** 2 - (ctable ** 2).sum(axis=1))

    stab /= n ** 2
    stabk = np.array([np.nan] * max(krange))
    for k in krange:
        stabk[k - 1] = np.mean(stab[:, k - 1])

    kopt = np.nanargmin(stabk) + 1

    out = {"kopt": kopt, "stabk": stabk, "stab": stab}
    return out
