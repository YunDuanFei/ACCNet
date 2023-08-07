import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.ticker as mtick
mpl.rcParams['agg.path.chunksize'] = 10000
sns.set_style("darkgrid")


def pt(p, t, name, ax):
    pmin, pmax = min(p), max(p)
    tmin, tmax = min(t), max(t)
    def jud(x, y):
        k = 1
        b = 0
        y_ = k * x + b
        return y >= y_
    ax.plot([0, 1], [0, 1], c='#040000', linestyle='--', linewidth=1, label='y=x')
    up_p, up_t = [], []
    do_p, do_t = [], []
    for x, y in zip(t, p):
        sta = jud(x, y)
        if sta:
            up_p.append(y)
            up_t.append(x)
        else:
            do_p.append(y)
            do_t.append(x)
    ax.scatter(up_t, up_p, s=0.5, c='#348498', marker='o', alpha=0.8, linewidths=0)
    ax.scatter(do_t, do_p, s=0.5, c='#ff502f', marker='o', alpha=0.8, linewidths=0)
    ax.set_xlim([-0, 1])
    ax.set_ylim([-0, 1])
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.set_xticks([0.01, 0.5, 0.9])
    ax.set_xticklabels(['0.0', '0.5', '1.0'])
    ax.set_yticks([0.01, 0.2, 0.4, 0.6, 0.8, 0.95])
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])

# prediction and true
def pretru(recor_path, i, fig, axs):
    print(recor_path)
    dfs = pd.read_csv(recor_path, encoding='UTF-8')
    recor_path = Path(recor_path)
    # for ahead
    min5t = np.asarray(dfs['min5t'].values[:])
    min5p = np.asarray(dfs['min5p'].values[:])
    axmin5 = axs[i, 0]
    pt(min5p, min5t, '5min-ahead', axmin5)

    min15t = np.asarray(dfs['min15t'].values[:])
    min15p = np.asarray(dfs['min15p'].values[:])
    axmin15 = axs[i, 1]
    pt(min15p, min15t, '15min-ahead', axmin15)

    min30t = np.asarray(dfs['min30t'].values[:])
    min30p = np.asarray(dfs['min30p'].values[:])
    axmin30 = axs[i, 2]
    pt(min30p, min30t, '30min-ahead', axmin30)

    h1t = np.asarray(dfs['h1t'].values[:])
    h1p = np.asarray(dfs['h1p'].values[:])
    axh1 = axs[i, 3]
    pt(h1p, h1t, '1h-ahead', axh1)

    h6t = np.asarray(dfs['h6t'].values[:])
    h6p = np.asarray(dfs['h6p'].values[:])
    axh6 = axs[i, 4]
    pt(h6p, h6t, '6h-ahead', axh6)

    d1t = np.asarray(dfs['d1t'].values[:])
    d1p = np.asarray(dfs['d1p'].values[:])
    axd1 = axs[i, 5]
    pt(d1p, d1t, '1d-ahead', axd1)

def make_files(path):
    filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            filelist.append(os.path.join(home, filename))
    nrows = len(filelist)
    ncols = 6
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, dpi=300, figsize=(18, 18))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.01, hspace=0.01)
    for i, file in enumerate(filelist):
        pretru(file, i, fig, axs)
    plt.draw()
    plt.savefig(path + '/' + "{}.tiff".format('CIGS'))
    plt.close(fig)


if __name__ == '__main__':
    recor_path = './vmd/norm_select_218-Site_9A-Solibro'
    make_files(recor_path)