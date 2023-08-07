import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections import OrderedDict
# change tdo and tup


def Pb(p, tdo, tup, pinc=0.9):
    ns = len(p)
    nssuml = 0.
    nssumu = 0.
    def pbl_i(p_i, tdo_i, pinc):
        if p_i >= tdo_i:
            nssuml_i = pinc / 2. * abs(p_i - tdo_i)
        else:
            nssuml_i = (1 - pinc / 2.) * abs(p_i - tdo_i)
        return nssuml_i
    def pbu_i(p_i, tup_i, pinc):
        if p_i >= tup_i:
            nssumu_i = (1 - pinc / 2.) * abs(p_i - tup_i)
        else:
            nssumu_i = pinc / 2. * abs(p_i - tup_i)
        return nssumu_i
    for p_i, tdo_i, tup_i in zip(p, tdo, tup):
        if np.isnan(tdo_i) or np.isnan(tup_i):
            tdo_i, tup_i = 0., 0.
        nssuml += pbl_i(p_i, tdo_i, pinc)
        nssumu += pbu_i(p_i, tup_i, pinc)
    return (1. / ns) * nssuml, (1. / ns) * nssumu

def Is(p, tdo, tup, pinc=0.9):
    ns = len(p)
    nssum = 0.
    def is_i(p_i, tdo_i, tup_i, pinc):
        if p_i < tdo_i:
            nssum_i = -2 * pinc * (tup_i - tdo_i) - 4 * (tdo_i - p_i)
        elif p_i > tup_i:
            nssum_i = -2 * pinc * (tup_i - tdo_i) - 4 * (p_i - tup_i)
        else:
            nssum_i = -2 * pinc * (tup_i - tdo_i)
        return nssum_i
    for p_i, tdo_i, tup_i in zip(p, tdo, tup):
        if np.isnan(tdo_i) or np.isnan(tup_i):
            tdo_i, tup_i = 0., 0.
        now = is_i(p_i, tdo_i, tup_i, pinc)
        nssum += now
    return (1. / ns) * nssum

def Ace(p, tdo, tup, pinc=0.9):
    ns = len(p)
    nssum = 0.
    def ace_i(p_i, tdo_i, tup_i):
        nssum_i = 1 if p_i >= tdo_i and p_i <= tup_i else 0
        return nssum_i
    for p_i, tdo_i, tup_i in zip(p, tdo, tup):
        if np.isnan(tdo_i) or np.isnan(tup_i):
            tdo_i, tup_i = 0., 0.
        nssum += ace_i(p_i, tdo_i, tup_i)
    return (1. / ns) * nssum - pinc

def cau_aip(path, record):
    path = Path(path)
    model = path.parent.parent.stem
    dataset = path.parent.parent.parent.stem
    print(model, dataset)
    dfs = pd.read_csv(path)
    min5p = np.asarray(dfs['min5p'].values[:])
    min5tup = np.asarray(dfs['min5tup'].values[:])  # 下界
    min5tdo = np.asarray(dfs['min5tdo'].values[:])  # 上界
    min5_num = len(min5tup)
    min5p = min5p[:min5_num]
    min5ace = Ace(min5p, min5tup, min5tdo)
    min5is = Is(min5p, min5tup, min5tdo)
    min5pbl, min5pbu = Pb(min5p, min5tup, min5tdo)
    print('min5:\n', min5ace, min5is, min5pbl, min5pbu)

    min15p = np.asarray(dfs['min15p'].values[:])
    min15tup = np.asarray(dfs['min15tup'].values[:])
    min15tdo = np.asarray(dfs['min15tdo'].values[:])
    min15_num = len(min15tup)
    min15p = min15p[:min15_num]
    min15ace = Ace(min15p, min15tup, min15tdo)
    min15is = Is(min15p, min15tup, min15tdo)
    min15pbl, min15pbu = Pb(min15p, min15tup, min15tdo)
    print('min15:\n', min15ace, min15is, min15pbl, min15pbu)

    min30p = np.asarray(dfs['min30p'].values[:])
    min30tup = np.asarray(dfs['min30tup'].values[:])
    min30tdo = np.asarray(dfs['min30tdo'].values[:])
    min30_num = len(min30tup)
    min30p = min30p[:min30_num]
    min30ace = Ace(min30p, min30tup, min30tdo)
    min30is = Is(min30p, min30tup, min30tdo)
    min30pbl, min30pbu = Pb(min30p, min30tup, min30tdo)
    print('min30:\n', min30ace, min30is, min30pbl, min30pbu)

    h1p = np.asarray(dfs['h1p'].values[:])
    h1tup = np.asarray(dfs['h1tup'].values[:])
    h1tdo = np.asarray(dfs['h1tdo'].values[:])
    h1_num = len(h1tup)
    h1p = h1p[:h1_num]
    h1ace = Ace(h1p, h1tup, h1tdo)
    h1is = Is(h1p, h1tup, h1tdo)
    h1pbl, h1pbu = Pb(h1p, h1tup, h1tdo)
    print('h1:\n', h1ace, h1is, h1pbl, h1pbu)

    h6p = np.asarray(dfs['h6p'].values[:])
    h6tup = np.asarray(dfs['h6tup'].values[:])
    h6tdo = np.asarray(dfs['h6tdo'].values[:])
    h6_num = len(h6tup)
    h6p = h6p[:h6_num]
    h6ace = Ace(h6p, h6tup, h6tdo)
    h6is = Is(h6p, h6tup, h6tdo)
    h6pbl, h6pbu = Pb(h6p, h6tup, h6tdo)
    print('h6:\n', h6ace, h6is, h6pbl, h6pbu)

    d1p = np.asarray(dfs['d1p'].values[:])
    d1tup = np.asarray(dfs['d1tup'].values[:])
    d1tdo = np.asarray(dfs['d1tdo'].values[:])
    d1_num = len(d1tup)
    d1p = d1p[:d1_num]
    d1ace = Ace(d1p, d1tup, d1tdo)
    d1is = Is(d1p, d1tup, d1tdo)
    d1pbl, d1pbu = Pb(d1p, d1tup, d1tdo)
    print('d1:\n', d1ace, d1is, d1pbl, d1pbu)

    record['dataset'] = record['dataset'] + [dataset] * 6
    record['model'] = record['model'] + [model] * 6
    record['time'] = record['time'] + ['min5', 'min15', 'min30', 'h1', 'h6', 'd1']
    record['ACE'] = record['ACE'] + [min5ace, min15ace, min30ace, h1ace, h6ace, d1ace]
    record['IS'] = record['IS'] + [min5is, min15is, min30is, h1is, h6is, d1is]
    record['PBL'] = record['PBL'] + [min5pbl, min15pbl, min30pbl, h1pbl, h6pbl, d1pbl]
    record['PBU'] = record['PBU'] + [min5pbu, min15pbu, min30pbu, h1pbu, h6pbu, d1pbu]

def make_files(path):
    record = OrderedDict(dataset=[], model=[], time=[], ACE=[], IS=[], PBL=[], PBU=[])
    filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.csv'):
                filelist.append(os.path.join(home, filename))
    for i, file in enumerate(filelist):
        cau_aip(file, record)
    path = Path(path)
    spath = path.parent / 'all_api'
    spath.mkdir(parents=True, exist_ok=True)
    csvpath = spath / 'all_api.csv'
    record = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in record.items()]))
    record.to_csv(csvpath, index=False)

if __name__ == '__main__':
    path = './api/vmd'
    make_files(path)






