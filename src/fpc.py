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
                krange=range(2, 10), count=False,
                nnk=1, largeisgood=False, **kargs):

    if clustermethod == 'kmeans':
        model = KMeans
    elif clustermethod == 'specc':
        model = SpectralClustering
    else:
        raise ValueError('Invalid clustermethod')
    classes = range(krange.start, krange.stop + 1)
    col_num = classes.stop - classes.start
    data = pd.DataFrame(data)
    n = data.shape[0]

    stab = np.zeros((B, col_num))
    for i, k in enumerate(classes):
        if count:
            print(f"{k} clusters")

        for b in range(B):
            if count:
                print(b)

            d1 = np.random.choice(n, n, replace=True)
            d2 = np.random.choice(n, n, replace=True)

            dmat1 = data.iloc[d1, :]
            dmat2 = data.iloc[d2, :]

            clm1 = model(n_clusters=k, **kargs).fit(dmat1)
            clm2 = model(n_clusters=k, **kargs).fit(dmat2)

            cj1 = pd.Series(-1, index=range(n))
            cj2 = pd.Series(-1, index=range(n))

            cj1[d1] = clm1.labels_
            cj2[d2] = clm2.labels_

            cj1 = classifnp(data, cj1, method=classification, centroids=None, nnk=nnk)
            cj2 = classifnp(data, cj2, method=classification, centroids=None, nnk=nnk)

            ctable = np.array(pd.crosstab(cj1, cj2))
            nck1 = ctable.sum(axis=1)
            stab[b, i] = np.sum(nck1 ** 2 - (ctable ** 2).sum(axis=1))

    stab /= n ** 2
    stab_mean = np.zeros(col_num)
    stab_std = np.zeros(col_num)
    for k in range(col_num):
        stab_mean[k] = np.mean(stab[:, k])
        stab_std[k] = np.std(stab[:, k])

    kopt = list(classes)[np.argmin(stab_mean)]

    out = {"kopt": kopt, "stab_mean" : stab_mean, "stab_std" : stab_std, "stab": stab}
    return out
