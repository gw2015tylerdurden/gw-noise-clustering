import hydra
import numpy as np
import pandas as pd
import umap
import src.fpc as fpc
from src.utils import calc_selfturining_affinity
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

    def log(self, **kwargs):
        wandb.log(kwargs)


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
    wdb = WandbLogging(args)
    np.random.seed(args.random_state)

    gravity_spy_labels = pd.read_csv(args.gravity_spy_path, index_col=0, header=None)
    z_autoencoder = pd.read_csv(args.dataset_path, index_col=0)

    z_umap = umap.UMAP(n_components=args.umap.n_components,
                       n_neighbors=args.umap.n_neighbors,
                       min_dist=args.umap.min_dist,
                       random_state=args.random_state).fit_transform(z_autoencoder)

    cs = fpc.ClusteringStability(args.sc.is_self_turning, args.sc.self_turning_neighbor)
    ret = cs.nselectboot(z_umap, B=args.bootstrap_num, krange=range(args.sc.n_start, args.sc.n_end), clustermethod="specc", count=True)

    error_rate = []
    for k in range(args.sc.n_start,  args.sc.n_end + 1):
        if args.sc.is_self_turning:
            sc = SpectralClustering(n_clusters=k, random_state=args.random_state, affinity='precomputed').fit(calc_selfturining_affinity(z_umap, args.sc.self_turning_neighbor))
        else:
            sc = SpectralClustering(n_clusters=k, random_state=args.random_state, gamma=args.sc.gamma).fit(z_umap)
        error_rate.append(calc_classification_error(gravity_spy_labels, sc.labels_))

    idx = 0
    for i in range(args.sc.n_end + 1):
        if i < args.sc.n_start:
            wdb.update(mean_stabilities=float('nan'), std=float('nan'), error_rate=float('nan'))
        else:
            wdb.update(mean_stabilities=ret["stab_mean"][idx], std=ret["stab_std"][idx], error_rate=error_rate[idx])
            idx += 1
    #wandb.log({'optimal class' : ret["kopt"]})


if __name__ == "__main__":
    main()
