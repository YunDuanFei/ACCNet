import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from norm_time import time_features
import os
import warnings
warnings.filterwarnings('ignore')
np.random.seed(0)
sns.set_style("darkgrid")
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000


def max_min(df):
    min = np.asarray(df).min()
    max = np.asarray(df).max()
    return [min, max]

def normalization(df, eps=1e-8):
    dics = {}
    max_min_scaler = lambda x : (x - np.min(x)) / (np.max(x) - np.min(x) + eps)
    df_stamp = df[['timestamp']]
    df_stamp['timestamp'] = pd.to_datetime(df_stamp.timestamp)
    df_stamp['year'] = df_stamp['timestamp'].dt.strftime('%Y')
    df_stamp['year'] = pd.to_numeric(df_stamp['year'])
    dics['year'] = max_min(df_stamp[['year']])
    df_nor = df_stamp[['year']].apply(max_min_scaler)
    nor_others = time_features(df_stamp, timeenc=1, freq='5min')
    dics['month'] = [11, 1]
    df_nor['month'] = nor_others[:, 3]
    dics['day'] = [30, 1]
    df_nor['day'] = nor_others[:, 2]
    dics['hour'] = [23, 0]
    df_nor['hour'] = nor_others[:, 1]
    dics['min'] = [59, 0]
    df_nor['min'] = nor_others[:, 0]
    dics['Active_Power'] = max_min(df[['Active_Power']])
    df_nor['AP'] = df[['Active_Power']].apply(max_min_scaler)
    dics['Wind_Speed'] = max_min(df[['Wind_Speed']])
    df_nor['WS'] = df[['Wind_Speed']].apply(max_min_scaler)
    dics['Weather_Temperature_Celsius'] = max_min(df[['Weather_Temperature_Celsius']])
    df_nor['WTC'] = df[['Weather_Temperature_Celsius']].apply(max_min_scaler)
    dics['Weather_Relative_Humidity'] = max_min(df[['Weather_Relative_Humidity']])
    df_nor['WRH'] = df[['Weather_Relative_Humidity']].apply(max_min_scaler)
    dics['Global_Horizontal_Radiation'] = max_min(df[['Global_Horizontal_Radiation']])
    df_nor['GHR'] = df[['Global_Horizontal_Radiation']].apply(max_min_scaler)
    dics['Diffuse_Horizontal_Radiation'] = max_min(df[['Diffuse_Horizontal_Radiation']])
    df_nor['DHR'] = df[['Diffuse_Horizontal_Radiation']].apply(max_min_scaler)
    dics['Wind_Direction'] = max_min(df[['Wind_Direction']])
    df_nor['WD'] = df[['Wind_Direction']].apply(max_min_scaler)
    dics['Weather_Daily_Rainfall'] = max_min(df[['Weather_Daily_Rainfall']])
    df_nor['WDR'] = df[['Weather_Daily_Rainfall']].apply(max_min_scaler)
    dics['Radiation_Global_Tilted'] = max_min(df[['Radiation_Global_Tilted']])
    df_nor['RGT'] = df[['Radiation_Global_Tilted']].apply(max_min_scaler)
    dics['Radiation_Diffuse_Tilted'] = max_min(df[['Radiation_Diffuse_Tilted']])
    df_nor['RDT'] = df[['Radiation_Diffuse_Tilted']].apply(max_min_scaler)

    return df_nor, dics

def error_correction(path):
    df = pd.read_csv(path)
    # print(df.head(4))
    # print(df.info())
    rows, cols = df.shape
    # print(rows, cols)
    for col in tqdm(range(1, cols), desc='proposing cols', leave=False):
        for row in tqdm(range(rows), leave=False):
            if np.isnan(df.iloc[row, col]):
                if row == 0:
                    if not np.isnan(df.iloc[row + 1, col]):
                        df.iloc[row, col] = df.iloc[row + 1, col]
                    else:
                        df.iloc[row, col] = 0
                elif row == rows - 1:
                    df.iloc[row, col] = df.iloc[row - 1, col]
                elif np.isnan(df.iloc[row + 1, col]):
                    df.iloc[row, col] = (df.iloc[row - 1, col] + 0) / 2
                else:
                    df.iloc[row, col] = (df.iloc[row-1, col] + df.iloc[row+1, col]) / 2
            else:
                pass
    path = Path(path)
    df.to_csv(path.parent / ('noout_' + path.name), index=0)
    df_nor, dics = normalization(df)
    df_nor.to_csv(path.parent/('norm_' + path.name), index=0)

    return df_nor, dics

def pltfig(path, dics):
    path = Path(path)
    norpath = Path(path).parent / ('norm_' + path.name)
    df_nor = pd.read_csv(norpath)
    names = [column for column in df_nor]
    for name in names:
        plt.figure(dpi=300, figsize=(24, 12))
        x = [i for i in range(len(df_nor[name]))]
        inter = 1 if name == "year" else 20
        inter_x, inter_y = x[::inter], df_nor[name].values[::inter]
        plt.plot(inter_x, inter_y, 'c', label='Active power')
        plt.ylabel('{}'.format(name))
        plt.xlabel('Time(Min)')
        plt.xlim(0, len(inter_x))
        plt.ylim(0, max(df_nor[name].values))
        plt.draw()
        dir_path = path.parent / path.stem
        dir_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(dir_path / '{}.tiff'.format(name))
        # plt.show()
    pathtxt = dir_path / '{}.txt'.format('raw_min_max_etc')
    with open(pathtxt, 'w') as f:
        for key, value in dics.items():
            f.write('Name: {} | one: {} | two: {}\n'.format(key, *value))

def check(df):
    if np.any(df.isnull()):
        msg = f"""
    Unsupported frequency NULL values
    """
        raise RuntimeError()

if __name__ == '__main__':
    path = './data_row/select_218-Site_9A-Solibro.csv'
    df_nor, dics = error_correction(path)
    check(df_nor)
    pltfig(path, dics)
