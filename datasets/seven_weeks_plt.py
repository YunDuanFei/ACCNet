import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
np.random.seed(0)
sns.set_style("darkgrid")
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000


def dfsw(path):
    df_nor = pd.read_csv(path)
    path = Path(path)
    norpath = Path(path).parent / path.stem
    norpath.mkdir(parents=True, exist_ok=True)
    names = [column for column in df_nor]
    for name in names:
        plt.figure(dpi=300, figsize=(8, 8))
        y = df_nor[name].values[:288*7]
        x = [i for i in range(len(y))]
        plt.plot(x, y, 'c')
        plt.ylabel('{}'.format(name))
        plt.xlabel('Time(Min)')
        plt.xlim(0, len(x))
        plt.ylim(0, max(y))
        plt.draw()
        plt.savefig(norpath / '{}.tiff'.format(name))


if __name__ == '__main__':
    path = './normdata/data_row/norm_select_218-Site_9A-Solibro.csv'
    dfsw(path)
