import hydra
import numpy as np
import pandas as pd
import umap
import src.fpc as fpc
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
        wandb.log(dict(kwargs))


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

@hydra.main(config_path="config", config_name="parameters")
def main(args):
    #wdb = WandbLogging(args)
    np.random.seed(args.random_state)

    gravity_spy_labels = pd.read_csv(args.gravity_spy_path, index_col=0, header=None)
    z_autoencoder = pd.read_csv(args.dataset_path, index_col=0)

    # z_umap = umap.UMAP(n_components=args.umap.n_components,
    #                    n_neighbors=args.umap.n_neighbors,
    #                    min_dist=args.umap.min_dist,
    #                    random_state=args.random_state).fit_transform(z_autoencoder)

    z_umap = pd.read_csv('/home/tyler/workspace/gw-noise-clustering/R/data/z_umap.csv', header=None)

    ret = fpc.nselectboot(z_umap, B=1, krange=range(args.sc.n_start, 24), clustermethod="specc", count=True, gamma=args.sc.gamma)
    #wdb.update(kopt=ret["kopt"], mean=ret["stabk"], std=np.std(ret["stab"], axis=1))
    print(ret)

    #sc = SpectralClustering(n_clusters=ret["kopt"], random_state=args.random_state, gamma=args.sc.gamma).fit(z_umap)
    #error_rate = calc_classification_error(gravity_spy_labels, sc.labels_)
    #wdb.update(error_rate=error_rate)

if __name__ == "__main__":
    main()
