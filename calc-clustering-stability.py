import hydra
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import SpectralClustering
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from collections import defaultdict, abc
import wandb

class WandbLogging:
    def __init__(self, args):
        self.stats = defaultdict(list)
        wandb.init(project=args.wandb.project, group=args.wandb.group)
        wandb.run.name = args.wandb.name + "_" + wandb.run.name
        wandb.config.update(self.__flatten(args))

    def __flatten(self, d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, abc.MutableMapping):
                items.extend(self.__flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.stats[key].append(value)

def calc_classification_error(true, pred):
    mclust = rpackages.importr('mclust')
    classification_error = robjects.r('classError')

    rtrue = numpy2ri.py2rpy(true)
    rpred = numpy2ri.py2rpy(pred)
    res = classification_error(rtrue, rpred)
    error_rate = res.rx2('errorRate')[0]
    return error_rate

@hydra.main(config_path="config", config_name="parameters")
def main(args):
    wdb = WandbLogging(args)

    label_indies = pd.read_csv(args.gravity_spy_path, index_col=0, header=None)
    z_autoencoder = pd.read_csv(args.dataset_path, index_col=0)

    z_umap = umap.UMAP(n_components=args.umap.n_components,
                       n_neighbors=args.umap.n_neighbors,
                       min_dist=args.umap.min_dist,
                       random_state=args.random_state).fit_transform(z_autoencoder)

    sc = SpectralClustering(n_clusters=26, random_state=args.random_state, assign_labels="kmeans", affinity="nearest_neighbors", n_neighbors=7).fit(z_umap)


    # classification_error関数を呼び出す
    mclust = rpackages.importr('mclust')
    classification_error = robjects.r('classError')

    rlabel_indies = numpy2ri.py2rpy(label_indies.values)
    rsc_labels = numpy2ri.py2rpy(sc.labels_)
    res = classification_error(rlabel_indies, rsc_labels)
    error_rate = res.rx2('errorRate')[0]
    wdb.update(error_rate)

    # clustering stability
    fpc = rpackages.importr('fpc')
    nselectboot = robjects.r('nselectboot')
    from rpy2.robjects.vectors import IntVector
    nclusters = nselectboot(data=numpy2ri.py2rpy(z_umap), B=50, clustermethod=fpc.speccCBI, classification="averagedist", krange=IntVector(range(args.sc.n_clusters)), count=True)
    print(nclusters)

if __name__ == "__main__":
    main()
