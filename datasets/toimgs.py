import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.vmd import VMD
from utils.dwt import DWT
import cv2
import os
from pathlib import Path
np.random.seed(0)
maxtimes = {
    '1d': 12*24,
    '6h': 12*6,
    '1h': 12,
    '30min': 6,
    '15min': 3,
    '5min': 1
}
choice_pearson = ['year', 'month', 'day', 'hour', 'min', 'AP', 'DHR', 'GHR', 'RDT', 'RGT', 'WTC']

def readcsv(path):
    df = pd.read_csv(path)
    choice_df = df[choice_pearson]

    path = Path(path)
    dataset_name = path.stem
    return choice_df, dataset_name

def signaltohsv(d_final, i, dname, path='./imgs/vmd', imgsets='train'):
    libpath = Path(os.path.join(path, dname, imgsets))
    libpath.mkdir(parents=True, exist_ok=True)
    norm_img = np.zeros(d_final.shape)
    norm_img = cv2.normalize(d_final, norm_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)  # 注意此处的三通道热力图是cv2专有的GBR排列
    img_hsv = cv2.cvtColor(heat_img, cv2.COLOR_BGR2HSV)
    img_hsv = cv2.resize(img_hsv, dsize=(72, 72))
    # img_path = os.path.join(path, dname, imgsets, 'vmd_{}.png'.format(str(i)))
    img_path = path + '/' + dname + '/' + imgsets + '/' + 'vmd_{}.png'.format(str(i))
    cv2.imwrite(img_path, img_hsv)
    return 'vmd_{}.png'.format(str(i))

def wrilabels(labels, maxtime, dname, path, imgsets):
    libpath = Path(os.path.join(path, dname))
    libpath.mkdir(parents=True, exist_ok=True)
    pathtxt = os.path.join(path, dname, imgsets + '_' + dname + '.txt')
    with open(pathtxt, 'w') as f:
        f.write('---------------------------Max Time {}|{}---------------------------\n'.format(imgsets, maxtime))
        for i, entry in enumerate(labels):
            f.write('name: {} | 1dAP: {} | 6hAP: {} | 1hAP: {} | 30minAP: {} | 15minAP: {} | 5minAP: {}\n'.format(*entry))

def to_vmd(dfs, dname, points=288, maxtime='1d', target='AP', svmd=3):
    assert maxtime == '1d'
    maxtime = maxtimes[maxtime]
    rows, cols = dfs.shape
    ssvmd = int(rows / svmd)
    names = [column for column in dfs]
    vmdds, train_labels, test_labels  = {}, [], []
    for name in names[5:]:  # drop year, month, day, hour, min ==>> called ymdh
        k = 1 if name != target else 3
        for i in range(svmd):
            if i == svmd - 1:
                vmdd, _, _ = VMD(dfs[name].values[ssvmd * i:], alpha=100, tau=0, K=k, DC=0, init=1, tol=1e-7)  # k x all points
            else:
                vmdd, _, _ = VMD(dfs[name].values[ssvmd*i:ssvmd*(i+1)], alpha=100, tau=0, K=k, DC=0, init=1, tol=1e-7)  # k x all points
            hvmdd = vmdd if i == 0 else np.hstack((hvmdd, vmdd))
        vmdds[name] = hvmdd
    vmdnames = list(vmdds.keys())
    last_year = max(np.asarray(dfs['year'].values[:]))
    for i in tqdm(range(rows-maxtime-points+1)):
        d_final = dfs.iloc[i:i+points, 0:5].T  # 5xpoints
        imgsets = 'train' if dfs.iloc[i, 0] != last_year else 'test'
        for name in vmdnames:
            d = vmdds[name][:, i:i+points]
            d_final = np.vstack((d_final, d))
        img_path = signaltohsv(d_final, i, dname, './imgs/vmd', imgsets)
        d1AC = dfs[target].values[i + points + maxtimes['1d'] - 1]
        h6AC = dfs[target].values[i + points + maxtimes['6h'] - 1]
        h1AC = dfs[target].values[i + points + maxtimes['1h'] - 1]
        min30AC = dfs[target].values[i + points + maxtimes['30min'] - 1]
        min15AC = dfs[target].values[i + points + maxtimes['15min'] - 1]
        min5AC = dfs[target].values[i + points + maxtimes['5min'] - 1]
        if imgsets == 'train':
            train_labels.append([img_path, d1AC, h6AC, h1AC, min30AC, min15AC, min5AC])
        else:
            test_labels.append([img_path, d1AC, h6AC, h1AC, min30AC, min15AC, min5AC])
    wrilabels(train_labels, maxtime, dname, './imgs/vmd', 'train')
    wrilabels(test_labels, maxtime, dname, './imgs/vmd', 'test')

def to_dwt(dfs, dname, points=72, maxtime='h', target='AP'):
    assert maxtime == 'h'
    maxtime = maxtimes[maxtime]
    rows, cols = dfs.shape
    names = [column for column in dfs][5:]
    train_labels, test_labels = [], []
    last_year = max(np.asarray(dfs['year'].values[:]))
    for i in tqdm(range(rows-maxtime-points+1)):
        d_final = dfs.iloc[i:i + points, 0:5].T  # 5xpoints
        imgsets = 'train' if dfs.iloc[i, 0] != last_year else 'test'
        ddwts = []
        for name in names:
            ddwts.append(dfs[name].values[i:i+points])  # (points,)
        img_path = DWT(ddwts, d_final, points, i, dname, imgsets)
        hAC = dfs[target].values[i + points + maxtime - 1]
        min30AC = dfs[target].values[i + points + maxtimes['30min'] - 1]
        min5AC = dfs[target].values[i + points + maxtimes['5min'] - 1]
        if imgsets == 'train':
            train_labels.append([img_path, hAC, min30AC, min5AC])
        else:
            test_labels.append([img_path, hAC, min30AC, min5AC])
    wrilabels(train_labels, maxtime, dname, './imgs/dwt', 'train')
    wrilabels(test_labels, maxtime, dname, './imgs/dwt', 'test')



if __name__ == '__main__':
    path = './normdata/data_row/norm_select_218-Site_9A-Solibro.csv'
    dfs, dname = readcsv(path)
    to_vmd(dfs, dname)
    # to_dwt(dfs, dname)
