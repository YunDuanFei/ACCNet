import numpy as np
import pywt
import cv2
from pathlib import Path
import os


def DWT(ddwts, d_final, num, i, dname, imgsets, wavelet='db1', mode='symmetric', maxlevel=3, path='./imgs/dwt'):
    d_final = np.asarray(d_final)
    libpath = Path(os.path.join(path, dname, imgsets))
    libpath.mkdir(parents=True, exist_ok=True)
    for j, d in enumerate(ddwts):
        wp = pywt.WaveletPacket(data=d, wavelet=wavelet, mode=mode, maxlevel=maxlevel)
        # 频谱由低到高的对应关系，这里需要注意小波变换的频带排列默认并不是顺序排列，所以这里需要使用’freq‘排序。
        freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
        # 计算maxlevel最小频段的带宽
        freqBand = num / (2 ** maxlevel)
        data_dwt = []
        for dec_coe in freqTree:
            dc_data = wp[dec_coe].data
            data_dwt.append(dc_data)
        data_dwt = np.asarray(data_dwt)
        data_dwts = data_dwt if j == 0 else np.vstack((data_dwts, data_dwt))
    normfinal_img = np.zeros(d_final.shape)
    normfinal_img = cv2.normalize(d_final, normfinal_img, 0, 255, cv2.NORM_MINMAX)
    normfinal_img = np.asarray(normfinal_img, dtype=np.uint8)
    heatfinal_img = cv2.applyColorMap(normfinal_img, cv2.COLORMAP_JET)  # 注意此处的三通道热力图是cv2专有的GBR排列
    heatfinal_hsv = cv2.cvtColor(heatfinal_img, cv2.COLOR_BGR2HSV)
    heatfinal_hsv = cv2.resize(heatfinal_hsv, dsize=(72, 72))
    normdwts_img = np.zeros(data_dwts.shape)
    normdwts_img = cv2.normalize(data_dwts, normdwts_img, 0, 255, cv2.NORM_MINMAX)
    normdwts_img = np.asarray(normdwts_img, dtype=np.uint8)
    heatdwts_img = cv2.applyColorMap(normdwts_img, cv2.COLORMAP_JET)  # 注意此处的三通道热力图是cv2专有的GBR排列
    heatdwts_hsv = cv2.cvtColor(heatdwts_img, cv2.COLOR_BGR2HSV)
    heatdwts_hsv = cv2.resize(heatdwts_hsv, dsize=(72, 72))
    imgs = np.vstack((heatfinal_hsv, heatdwts_hsv))
    img_path = path + '/' + dname + '/' + imgsets + '/' + 'vmd_{}.png'.format(str(i))
    cv2.imwrite(img_path, imgs)

    return img_path

