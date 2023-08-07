import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from utils.vmd import VMD
from matplotlib.pyplot import MultipleLocator
import warnings
warnings.filterwarnings('ignore')
np.random.seed(0)
sns.set_style("darkgrid")
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000

def pltfig(vmdd, norpath, long=288*7):
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
        plt.savefig(norpath / 'vmd_{}.tiff'.format(i), bbox_inches='tight')
        # plt.show()

def to_vmd(path):
    dfs = pd.read_csv(path)
    path = Path(path)
    norpath = Path(path).parent / path.stem
    norpath.mkdir(parents=True, exist_ok=True)
    df = dfs['AP'].values[:]
    width = len(df)
    vmdd, _, _ = VMD(df, alpha=100, tau=0, K=3, DC=0, init=1, tol=1e-7)
    df = np.asarray(df).reshape(1, -1)[:, :(width-1)]
    vmdd = np.vstack((df, vmdd))
    pltfig(vmdd, norpath)
    new_dfs = pd.DataFrame()
    imfs = ['ap_row', 'ap_imf1', 'ap_imf2', 'residual']
    for i, name in enumerate(imfs):
        new_dfs[name] = vmdd[i, :288*7]
    new_dfs['x'] = [i for i in range(0, 288*7)]
    pathtxt = os.path.join(norpath, 'vmd_seven.csv')
    new_dfs.to_csv(pathtxt, index=False)




if __name__ == '__main__':
    path = './normdata/data_row/norm_select_218-Site_9A-Solibro.csv'
    to_vmd(path)
