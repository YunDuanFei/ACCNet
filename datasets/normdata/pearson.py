import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib as mpl
sns.set_style('darkgrid')
mpl.rcParams['agg.path.chunksize'] = 100000


def pltfig(df, rho, prob, savep, target):
    x = np.asarray(df[target].values[:])
    for i, name in enumerate(prob):
        plt.figure(dpi=300, figsize=(6, 6))
        y = np.asarray(df[name].values[:])
        plt.scatter(x, y, s=0.5, c='#348498', marker='o', alpha=0.8, linewidths=0)
        plt.title('Correlation = ' + "{:.2f}".format(rho[0, i+1]))
        plt.xlabel('Active PV power (KW)')
        name = name.replace('_', ' ')
        plt.ylabel(name)
        plt.draw()
        plt.savefig(savep / "{}.tiff".format(target + '_' + name))

def caupearson(path, target):
    prob = ['Wind_Speed', 'Weather_Temperature_Celsius', 'Weather_Relative_Humidity', 'Global_Horizontal_Radiation',
            'Diffuse_Horizontal_Radiation', 'Wind_Direction', 'Weather_Daily_Rainfall', 'Radiation_Global_Tilted',
            'Radiation_Diffuse_Tilted']
    df = pd.read_csv(path)
    dtarget = np.asarray(df[target].values[:])
    for name in prob:
        dapro = np.asarray(df[name].values[:])
        dtarget = np.vstack((dtarget, dapro))
    rho = np.corrcoef(dtarget)
    path = Path(path)
    save_path = path.parent / path.stem
    save_path.mkdir(parents=True, exist_ok=True)
    pltfig(df, rho, prob, save_path, target)
    print(rho)


if __name__ == '__main__':
    path = './data_row/noout_select_218-Site_9A-Solibro.csv'
    target = "Active_Power"
    caupearson(path,target)

