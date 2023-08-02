# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 15:26:32 2023

@author: Joe
"""
from img_model import VGG16, build_model
from img_process import read_img
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

path = 'H:/Sort_1000pics/'

img_size = (128,128)

# 讀取影像
fpaths, data, label = read_img(path)
print(data.shape)  # (1000, 128, 128, 3)
# 計算有多少類圖片
classes = len(set(label))
print(classes)

# 生成等差數列隨機調整影像順序
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]
fpaths = fpaths[arr]

# 拆分訓練集和測試集 70%訓練集 30%測試集
ratio = 0.7
s = np.int32(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
fpaths_train = fpaths[:s] 
x_test = data[s:]
y_test = label[s:]
fpaths_test = fpaths[s:] 
print(len(x_train),len(y_train),len(x_test),len(y_test)) #700 700 300 300
print(y_test)

#------------------------------------------VGG16-------------------------------------

model = VGG16(data.shape[1:], 10)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, epochs=70, batch_size=64,validation_split=0.1)

loss, accuracy = model.evaluate(x_test, y_test)

model.save(path + 'VGG16_Model.h5')
#------------------------------------------ResNet50-------------------------------------

Adam=tf.compat.v1.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model = build_model(ResNet50, 10, data.shape[1:], Adam)

model.summary()

history = model.fit(x_train, y_train, epochs=200, batch_size=256,validation_split=0.1)

loss, accuracy = model.evaluate(x_test, y_test)

model.save(path + 'ResNet50_Model.h5') #儲存模型

#--------------------------------------讀取模型-----------------------------------------

model = tf.keras.models.load_model(path + 'ResNet50_Model.h5')

loss, accuracy = model.evaluate(x_test, y_test)