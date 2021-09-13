import matplotlib.pyplot as plt
import sys
from keras.preprocessing.image import array_to_img,img_to_array,load_img
import numpy as np
import os
from keras.preprocessing import image
import keras
from sklearn.model_selection import train_test_split

import load_image
import Train_metric_cnn
import Faster_Grad_CAM
import Train_metric_cnn

#GPUを使用するためのコード  
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="1", # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))

#ここを変えてから動かす
#クラスはここに記入
#MODEL_NAMEは最後尾に.h5がないとだめ
CLASS = 7
EPOCH = 100
MODEL_NAME = "model_label_7_100.json"
MODEL_WEIGHTS = "model_label_7_100.h5"
PATH = "model/"



#データセットの作成
x_train_0 = load_image.load_images("images/data/0/")
x_train_1 = load_image.load_images("images/data/1/")
x_train_2 = load_image.load_images("images/data/2/")
x_train_3 = load_image.load_images("images/data/3/")
x_train_4 = load_image.load_images("images/data/4/")
x_train_5 = load_image.load_images("images/data/5/")
x_train_6 = load_image.load_images("images/data/6/")
#x_train_7 = load_images("images/data/7/")
print(x_train_0)

x_train_0 /= 255
x_train_1 /= 255
x_train_2 /= 255
x_train_3 /= 255
x_train_4 /= 255
x_train_5 /= 255
x_train_6 /= 255
#x_train_7 /= 255

train_label_0 = np.full(len(x_train_0),0)
train_label_1 = np.full(len(x_train_1),1)
train_label_2 = np.full(len(x_train_2),2)
train_label_3 = np.full(len(x_train_3),3)
train_label_4 = np.full(len(x_train_4),4)
train_label_5 = np.full(len(x_train_5),5)
train_label_6 = np.full(len(x_train_6),6)
#train_label_7 = np.full(len(x_train_7),7)


x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(x_train_0, train_label_0, train_size=0.8)
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_train_1, train_label_1, train_size=0.8)
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_train_2, train_label_2, train_size=0.8)
x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(x_train_3, train_label_3, train_size=0.8)
x_train_4, x_test_4, y_train_4, y_test_4 = train_test_split(x_train_4, train_label_4, train_size=0.8)
x_train_5, x_test_5, y_train_5, y_test_5 = train_test_split(x_train_5, train_label_5, train_size=0.8)
x_train_6, x_test_6, y_train_6, y_test_6 = train_test_split(x_train_6, train_label_6, train_size=0.8)
#x_train_7, x_test_7, y_train_7, y_test_7 = train_test_split(x_train_7, train_label_7, train_size=0.8)

X_train = np.vstack((x_train_0,x_train_1,x_train_2,x_train_3,x_train_4,x_train_5,x_train_6))
X_test = np.vstack((x_test_0,x_test_1,x_test_2,x_test_3,x_test_4,x_test_5,x_test_6))
y_train = np.hstack((y_train_0,y_train_1,y_train_2,y_train_3,y_train_4,y_train_5,y_train_6))
y_test = np.hstack((y_test_0,y_test_1,y_test_2,y_test_3,y_test_4,y_test_5,y_test_6))

Y_train = keras.utils.to_categorical(y_train)
Y_test = keras.utils.to_categorical(y_test)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

#model作成
model = Train_metric_cnn.train_metric_cnn(X_train, Y_train, X_test, Y_test, classes=CLASS,PATH=PATH,MODEL_NAME=MODEL_NAME,MODEL_WEIGHTS=MODEL_WEIGHTS)
#学習開始
model.train_mobileV2(epoch=EPOCH)
#gradcamの学習
model_grad, kmeans, channel_weight, channel_adress, vector_normal = Faster_Grad_CAM.train_faster_gradcam(x_train_0,x_train_1,x_train_3, x_train_2, model)
#結果を表示gradcamとかを
Faster_Grad_CAM.show_result(model, model_grad, "block5_conv4", x_test_0, [0,1,2,3,4,5], kmeans, channel_weight, channel_adress)

