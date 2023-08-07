import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
sns.set_style("darkgrid")


def pt(p, t, spath, name):
    pmin, pmax = min(p), max(p)
    tmin, tmax = min(t), max(t)
    def jud(x, y):
        k = (pmax - pmin) / (tmax - tmin + 1e-8)
        b = pmin - k * tmin
        y_ = k * x + b
        return y >= y_
    plt.figure(dpi=300, figsize=(6, 6))
    plt.plot([tmin, tmax], [pmin, pmax], c='#040000', linestyle='--', linewidth=1)
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
    plt.scatter(up_t, up_p, s=0.5, c='#348498', marker='o', alpha=0.8, linewidths=0, label='over-predicted')
    plt.scatter(do_t, do_p, s=0.5, c='#ff502f', marker='o', alpha=0.8, linewidths=0, label='under-predicted')
    plt.title('{}'.format(name))
    plt.xlabel('Actual PV output (KW)')
    plt.ylabel('Predicted PV output (KW)')
    plt.draw()
    plt.savefig(spath / "{}.tiff".format(name))

# prediction and true
def pretru(recor_path):
    dfs = pd.read_csv(recor_path)
    recor_path = Path(recor_path)
    # for ahead
    trupre = Path(recor_path).parent / 'trupre'
    trupre.mkdir(parents=True, exist_ok=True)
    min5t = np.asarray(dfs['min5t'].values[:])
    min5p = np.asarray(dfs['min5p'].values[:])
    pt(min5p, min5t, trupre, '5min-ahead')
    min15t = np.asarray(dfs['min15t'].values[:])
    min15p = np.asarray(dfs['min15p'].values[:])
    pt(min15p, min15t, trupre, '15min-ahead')
    min30t = np.asarray(dfs['min30t'].values[:])
    min30p = np.asarray(dfs['min30p'].values[:])
    pt(min30p, min30t, trupre, '30min-ahead')
    h1t = np.asarray(dfs['h1t'].values[:])
    h1p = np.asarray(dfs['h1p'].values[:])
    pt(h1p, h1t, trupre, '1h-ahead')
    h6t = np.asarray(dfs['h6t'].values[:])
    h6p = np.asarray(dfs['h6p'].values[:])
    pt(h6p, h6t, trupre, '6h-ahead')
    d1t = np.asarray(dfs['d1t'].values[:])
    d1p = np.asarray(dfs['d1p'].values[:])
    pt(d1p, d1t, trupre, '1d-ahead')

def iv(t, p, tup, tdo, pup, pdo, spath, name, ise):
    x = [i for i in range(len(t))]
    plt.figure(dpi=300, figsize=(6, 6))
    plt.plot(x, t, c='#A2C5FD', linestyle='-', linewidth=1, label='Actual value')
    plt.plot(x, p, c='#FCB7A0', linestyle='-', linewidth=1, label='Predicted value')
    plt.fill_between(x, tup, tdo, alpha=0.5, color='#BAE0FB', label='Confidence interval')
    plt.fill_between(x, pup, pdo, alpha=0.5, color='#FFEAD0', label='Prediction interval')
    plt.legend()
    plt.title('{}'.format(name))
    plt.xlabel('Time')
    plt.ylabel('PV (KW)')
    plt.draw()
    plt.savefig(spath / "{}.tiff".format(name + '_' + str(ise)))
    plt.close()

# plt interval
def interval(recor_path, numd=50):
    day = 0
    dfs = pd.read_csv(recor_path)
    numpins = len(dfs['min5t'].values[:])
    recor_path = Path(recor_path)
    min5 = Path(recor_path).parent / '5min-ahead'
    min5.mkdir(parents=True, exist_ok=True)
    min15 = Path(recor_path).parent / '15min-ahead'
    min15.mkdir(parents=True, exist_ok=True)
    min30 = Path(recor_path).parent / '30min-ahead'
    min30.mkdir(parents=True, exist_ok=True)
    h1 = Path(recor_path).parent / '1h-ahead'
    h1.mkdir(parents=True, exist_ok=True)
    h6 = Path(recor_path).parent / '6h-ahead'
    h6.mkdir(parents=True, exist_ok=True)
    d1 = Path(recor_path).parent / '1d-ahead'
    d1.mkdir(parents=True, exist_ok=True)
    while day < numpins and day <= numd:
        min5t = np.asarray(dfs['min5t'].values[day * 288:(day + 1) * 288])
        min5p = np.asarray(dfs['min5p'].values[day * 288:(day + 1) * 288])
        min5tup = np.asarray(dfs['min5tup'].values[day * 288:(day + 1) * 288])
        min5tdo = np.asarray(dfs['min5tdo'].values[day * 288:(day + 1) * 288])
        min5pup = np.asarray(dfs['min5pup'].values[day * 288:(day + 1) * 288])
        min5pdo = np.asarray(dfs['min5pdo'].values[day * 288:(day + 1) * 288])
        iv(min5t, min5p, min5tup, min5tdo, min5pup, min5pdo, min5, '5min-ahead', day)

        min15t = np.asarray(dfs['min15t'].values[day * 288:(day + 1) * 288])
        min15p = np.asarray(dfs['min15p'].values[day * 288:(day + 1) * 288])
        min15tup = np.asarray(dfs['min15tup'].values[day * 288:(day + 1) * 288])
        min15tdo = np.asarray(dfs['min15tdo'].values[day * 288:(day + 1) * 288])
        min15pup = np.asarray(dfs['min15pup'].values[day * 288:(day + 1) * 288])
        min15pdo = np.asarray(dfs['min15pdo'].values[day * 288:(day + 1) * 288])
        iv(min15t, min15p, min15tup, min15tdo, min15pup, min15pdo, min15, '15min-ahead', day)

        min30t = np.asarray(dfs['min30t'].values[day * 288:(day + 1) * 288])
        min30p = np.asarray(dfs['min30p'].values[day * 288:(day + 1) * 288])
        min30tup = np.asarray(dfs['min30tup'].values[day * 288:(day + 1) * 288])
        min30tdo = np.asarray(dfs['min30tdo'].values[day * 288:(day + 1) * 288])
        min30pup = np.asarray(dfs['min30pup'].values[day * 288:(day + 1) * 288])
        min30pdo = np.asarray(dfs['min30pdo'].values[day * 288:(day + 1) * 288])
        iv(min30t, min30p, min30tup, min30tdo, min30pup, min30pdo, min30, '30min-ahead', day)

        h1t = np.asarray(dfs['h1t'].values[day * 288:(day + 1) * 288])
        h1p = np.asarray(dfs['h1p'].values[day * 288:(day + 1) * 288])
        h1tup = np.asarray(dfs['h1tup'].values[day * 288:(day + 1) * 288])
        h1tdo = np.asarray(dfs['h1tdo'].values[day * 288:(day + 1) * 288])
        h1pup = np.asarray(dfs['h1pup'].values[day * 288:(day + 1) * 288])
        h1pdo = np.asarray(dfs['h1pdo'].values[day * 288:(day + 1) * 288])
        iv(h1t, h1p, h1tup, h1tdo, h1pup, h1pdo, h1, '1h-ahead', day)

        h6t = np.asarray(dfs['h6t'].values[day * 288:(day + 1) * 288])
        h6p = np.asarray(dfs['h6p'].values[day * 288:(day + 1) * 288])
        h6tup = np.asarray(dfs['h6tup'].values[day * 288:(day + 1) * 288])
        h6tdo = np.asarray(dfs['h6tdo'].values[day * 288:(day + 1) * 288])
        h6pup = np.asarray(dfs['h6pup'].values[day * 288:(day + 1) * 288])
        h6pdo = np.asarray(dfs['h6pdo'].values[day * 288:(day + 1) * 288])
        iv(h6t, h6p, h6tup, h6tdo, h6pup, h6pdo, h6, '6h-ahead', day)

        d1t = np.asarray(dfs['d1t'].values[day * 288:(day + 1) * 288])
        d1p = np.asarray(dfs['d1p'].values[day * 288:(day + 1) * 288])
        d1tup = np.asarray(dfs['d1tup'].values[day * 288:(day + 1) * 288])
        d1tdo = np.asarray(dfs['d1tdo'].values[day * 288:(day + 1) * 288])
        d1pup = np.asarray(dfs['d1pup'].values[day * 288:(day + 1) * 288])
        d1pdo = np.asarray(dfs['d1pdo'].values[day * 288:(day + 1) * 288])
        iv(d1t, d1p, d1tup, d1tdo, d1pup, d1pdo, d1, '1d-ahead', day)

        day += 1


if __name__ == '__main__':
    recor_path = './vmd/norm_select_218-Site_9A-Solibro/lan/cap/norm_select_218-Site_9A-Solibro.csv'
    pretru(recor_path)
    interval(recor_path)
