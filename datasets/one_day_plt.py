import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.vmd import VMD
import os
from matplotlib.pyplot import MultipleLocator
import warnings
warnings.filterwarnings('ignore')
np.random.seed(0)
sns.set_style("darkgrid")
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000

def pltfig(vmdd, norpath, long=288*1):
    num = len(vmdd)
    for i in range(num):
        if i == 0:
            name = 'Original AP'
        elif i < num-1:
            name = 'IMF' + str(i)
        else:
            name = 'Residue'
        plt.figure(dpi=600, figsize=(8, 8))
        x = [j for j in range(len(vmdd[i, :]))]
        x, y = x[:long], vmdd[i, :long]
        plt.plot(x, y, 'c', label='Active power')
        plt.ylabel('{}'.format(name))
        plt.xlabel('Time(Min)')
        maxsize = 5
        m = 0.1
        N = len(x)
        s = maxsize / plt.gcf().dpi * N + 2 * m
        margin = m / plt.gcf().get_size_inches()[0]
        plt.gcf().subplots_adjust(left=margin, right=1. - margin)
        plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
        plt.xlim(0, len(x))
        plt.ylim(min(y), max(y))
        plt.draw()
        plt.savefig(norpath / 'one_day_vmd_{}.tiff'.format(i), bbox_inches='tight')
        # plt.show()

def dfsw(dfs, norpath):
    names = [column for column in dfs]
    for name in names:
        plt.figure(dpi=600, figsize=(8, 8))
        y = dfs[name].values[:288*1]
        x = [i for i in range(len(y))]
        plt.plot(x, y, 'c')
        plt.ylabel('{}'.format(name))
        plt.xlabel('Time(Min)')
        plt.xlim(0, len(x))
        plt.ylim(min(y), max(y))
        plt.draw()
        plt.savefig(norpath / 'one_day_{}.tiff'.format(name))

def to_vmd(path):
    dfs = pd.read_csv(path)
    path = Path(path)
    norpath = Path(path).parent / path.stem
    norpath.mkdir(parents=True, exist_ok=True)
    df = dfs['AP'].values[:]
    vmdd, _, _ = VMD(df, alpha=100, tau=0, K=3, DC=0, init=1, tol=1e-7)
    pathtxt = os.path.join(norpath, 'one_day.csv')
    names = [column for column in dfs]
    imfs = ['ap_imf1', 'ap_imf2', 'residual']
    new_dfs = pd.DataFrame()
    for name in names:
        new_dfs[name] = dfs.loc[:287, name]
    for i, name in enumerate(imfs):
        new_dfs[name] = vmdd[i, :288]
    new_dfs['x'] = [i for i in range(0, 288)]
    new_dfs.to_csv(pathtxt, index=False)



if __name__ == '__main__':
    path = './normdata/data_row/norm_select_218-Site_9A-Solibro.csv'
    to_vmd(path)
