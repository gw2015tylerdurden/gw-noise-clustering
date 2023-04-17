import numpy as np
import random
import pandas as pd
from .utils import calc_selfturining_affinity, calc_classification_error
from sklearn.cluster import KMeans, SpectralClustering
from scipy.spatial.distance import pdist, squareform


class SpeccClusteringInstability:
    def __init__(self, gamma):
        self.is_self_turining = False
        self.is_median_heuristic = False

        if isinstance(gamma, float):
            self.gamma = gamma
        elif gamma == 'median-heuristic':
            self.is_median_heuristic = True
        elif gamma.startswith('self-turning-neighbor'):
            self.self_turning_neighbor = int(gamma.split(':')[1])
            self.is_self_turining = True
        else:
            raise ValueError('Invalid gamma')

    def __get_bootstrap_pair(self, data, n):
            d1 = np.random.choice(n, n, replace=True)
            d2 = np.random.choice(n, n, replace=True)

            dmat1 = data.iloc[d1, :]
            dmat2 = data.iloc[d2, :]
            return dmat1, d1, dmat2, d2

    def __classifnp(self, data, clustering, method='centroid', cdist=None, centroids=None, nnk=1):
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


    def nselectboot(self, data, B=50, distances=None, clustermethod="kmeans",
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

                dmat1, d1, dmat2, d2 = self.__get_bootstrap_pair(data, n)

                if self.is_self_turining:
                    while(True):
                        X1, success1 = calc_selfturining_affinity(dmat1, self.self_turning_neighbor)
                        X2, success2 = calc_selfturining_affinity(dmat2, self.self_turning_neighbor)
                        if (success1 is True and success2 is True):
                            break
                        else:
                            print(f"Sigma has detected 0 for k-neighbor (set as {self.self_turning_neighbor}). Regenerate bootstrap pairs by changing random seed.")
                            np.random.seed(random.randint(0, 100000))
                            dmat1, d1, dmat2, d2 = self.__get_bootstrap_pair(data, n)

                    kargs['affinity'] = 'precomputed'
                    clm1 = model(n_clusters=k, **kargs).fit(X1)
                    clm2 = model(n_clusters=k, **kargs).fit(X2)
                elif self.is_median_heuristic:
                    kargs['gamma'] = self.get_medianheuristic_gamma(dmat1)
                    clm1 = model(n_clusters=k, **kargs).fit(dmat1)
                    kargs['gamma'] = self.get_medianheuristic_gamma(dmat2)
                    clm2 = model(n_clusters=k, **kargs).fit(dmat2)
                else:
                    kargs['gamma'] = self.gamma
                    clm1 = model(n_clusters=k, **kargs).fit(dmat1)
                    clm2 = model(n_clusters=k, **kargs).fit(dmat2)

                cj1 = pd.Series(-1, index=range(n))
                cj2 = pd.Series(-1, index=range(n))

                cj1[d1] = clm1.labels_
                cj2[d2] = clm2.labels_

                cj1 = self.__classifnp(data, cj1, method=classification, centroids=None, nnk=nnk)
                cj2 = self.__classifnp(data, cj2, method=classification, centroids=None, nnk=nnk)

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


    def get_medianheuristic_gamma(self, data):
        return 2.0 / np.median(pdist(data)) ** 2


    def get_error_rate(self, data, true, krange=(2, 10), seed=123):
        classes = range(krange.start, krange.stop + 1)

        error_rate = []
        for k in classes:
            if self.is_self_turining:
                X, _ = calc_selfturining_affinity(data, self.self_turning_neighbor)
                sc = SpectralClustering(n_clusters=k, random_state=seed, affinity='precomputed').fit(X)
            elif self.is_median_heuristic:
                gamma = self.get_medianheuristic_gamma(data)
                sc = SpectralClustering(n_clusters=k, random_state=seed, gamma=gamma).fit(data)
            else:
                sc = SpectralClustering(n_clusters=k, random_state=seed, gamma=self.gamma).fit(data)
            error_rate.append(calc_classification_error(true, sc.labels_))

        return error_rate

