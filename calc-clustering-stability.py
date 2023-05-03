import hydra
import numpy as np
import pandas as pd
import umap
import src.fpc as fpc
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

    cs = fpc.SpeccClusteringInstability(args.sc.gamma)
    ret = cs.nselectboot(z_umap, B=args.bootstrap_num, krange=range(args.sc.n_start, args.sc.n_end), clustermethod="specc", count=True, nnk=args.clustering_stability_nnk)
    error_rate = cs.get_error_rate(z_umap, gravity_spy_labels, krange=range(args.sc.n_start, args.sc.n_end), seed=args.random_state)

    idx = 0
    for i in range(args.sc.n_end + 1):
        if i < args.sc.n_start:
            wdb.update(mean_instabilities=float('nan'), std=float('nan'), error_rate=float('nan'))
        else:
            wdb.update(mean_instabilities=ret["stab_mean"][idx], std=ret["stab_std"][idx], error_rate=error_rate[idx])
            idx += 1
    # summary
    wandb.log({'min mean stability' : min(ret["stab_mean"]),  'min error rate' : min(error_rate)})


if __name__ == "__main__":
    main()
