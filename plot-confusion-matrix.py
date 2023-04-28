import numpy as np
import pandas as pd
import argparse
import seaborn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

gravity_spy_labels = ['1080Lines', '1400Ripples', 'Air_Compressor', 'Blip',
         'Chirp', 'Extremely_Loud', 'Helix', 'Koi_Fish',
         'Light_Modulation', 'Low_Frequency_Burst', 'Low_Frequency_Lines', 'No_Glitch',
         'None_of_the_Above', 'Paired_Doves', 'Power_Line', 'Repeating_Blips',
         'Scattered_Light', 'Scratchy', 'Tomte', 'Violin_Mode',
         'Wandering_Line', 'Whistle']


def plot_confusion_matrix(true_y, pred_y):
    true_y, _ = pd.factorize(true_y.iloc[:, 0])
    pred_y = pred_y.iloc[:, 0]
    true_class_num = len(np.unique(true_y))
    pred_class_num = len(np.unique(pred_y))
    cm_squared = confusion_matrix(true_y, pred_y)
    if true_class_num > pred_class_num:
        labels = true_class_num
        cm = cm_squared[:, :pred_class_num]
        cmap = 'Blues'
        ax_xlabel = "Unsupervised Labels - median-heuristic"
    else:
        labels = pred_class_num
        cm = cm_squared[:true_class_num, :]
        cmap = 'Greens'
        ax_xlabel = "Unsupervised Labels - selfturning"

    cmn = normalize(cm, norm='l1', axis=0)

    # plot
    fig, ax = plt.subplots(figsize=(15, 10))
    round_rate = np.round(cmn, decimals=2)
    annot_str = np.where(round_rate != 0, round_rate, '')

    seaborn.heatmap(
        data=cmn,
        ax=ax,
        annot=annot_str,
        annot_kws={"fontsize": 9.5},
        #fmt=".1f",
        fmt='',
        linewidths=0.1,
        #cmap="Greens",
        cmap=cmap,
        #cbar=False,
        #cbar_kws={"aspect": 50, "pad": 0.01, "anchor": (0, 0.05), "use_gridspec": False, "location": 'bottom'},
        #yticklabels=self.target_labels_name,
        yticklabels=gravity_spy_labels,
        xticklabels=np.arange(pred_class_num),
        #square=True,
    )
    ax.set_xlabel(ax_xlabel)
    ax.set_ylabel(r"Supervised Labels")

    plt.tight_layout()
    plt.savefig("cm_sc.svg", transparent=True, dpi=300)
    plt.show()
    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument('arg1', type=str, help='input csv file of pred labels')
args = parser.parse_args()

pred_labels = pd.read_csv(args.arg1, index_col=0, header=0)
true_labels = pd.read_csv('./data/z-autoencoder-outputs.csv', index_col=None)

plot_confusion_matrix(true_labels, pred_labels)
