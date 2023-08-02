# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 14:17:00 2023

@author: Joe
"""

import os
import glob
import cv2
import numpy as np

def read_img(path, img_size):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    fpath = []
    for idx, folder in enumerate(cate):
        # 遍歷整個目錄判斷每個檔案是不是符合
        for im in glob.glob(folder + '/*.jpg'):
            #print('reading the images:%s' % (im))
            img = cv2.imread(im)             #呼叫opencv庫讀取畫素點
            img = cv2.resize(img, img_size)  #影像畫素大小一致
            imgs.append(img)                 #影像資料
            labels.append(idx)               #影像類標
            fpath.append(path+im)            #影像路徑名
            #print(path+im, idx)
    return np.asarray(fpath, np.string_), np.asarray(imgs, np.float32), np.asarray(labels, np.int32)
