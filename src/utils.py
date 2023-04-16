import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import euclidean_distances
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri


def calc_selfturining_affinity(X, neighbor_num=7, duplicate_sigma=1.0):
    """Computes affinity matrix using self-turining method.
    Read more in the :ref:`https://proceedings.neurips.cc/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf`.

    Exsample code:
    # original
    sc = SpectralClustering(n_clusters=2, assign_labels="kmeans", affinity="rbf").fit(X)

    # self-turning
    sc = SpectralClustering(n_clusters=2, assign_labels="kmeans", affinity="precomputed").fit(calc_selfturining_affinity(X))
    """
    K = euclidean_distances(X, X, squared=True)

    # local scaling
    nn = NearestNeighbors(n_neighbors=neighbor_num + 1).fit(X)  # +1はターゲットのデータ分
    dist, indices = nn.kneighbors(X)
    sigmas = dist[:, -1]  # -1によりターゲットとneighbor_num番目間との距離を取得する
    if np.any(sigmas == 0):
        # 重複したデータセットの場合、近傍値が0になるため、duplicate_sigmaを設定
        sigmas = np.where(sigmas == 0, duplicate_sigma, sigmas)
    K = -1.0 * K / np.outer(sigmas, sigmas)
    affinity = np.exp(K)  # exponentiate K in-place
    return affinity


def calc_classification_error(true, pred):
    true = np.array(true)
    pred = np.array(pred)

    mclust = rpackages.importr('mclust')
    classification_error = robjects.r('classError')

    rtrue = numpy2ri.py2rpy(true)
    rpred = numpy2ri.py2rpy(pred)
    res = classification_error(rtrue, rpred)
    error_rate = res.rx2('errorRate')[0]
    return error_rate
