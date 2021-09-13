import time
from keras import backend as K
import tensorflow as tf
from PIL import Image
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

class GradCam():
    def __init__(self,model, x, layer_name, class_, top50):
        self.model = model
        self.x = x
        self.layer_name = layer_name
        self.class_ = class_
        self.top50 = top50

    def gradcam(self):
        X = np.expand_dims(self.x,axis=0)
        
        # 前処理
        target = np.array([0, 1]).reshape((1,-1))
        class_idx = self.class_
        class_output = self.model.output[:, class_idx]
        
        # 勾配を取得
        before = time.time()
        conv_output = self.model.get_layer(self.layer_name).output   # layer_nameのレイヤーのアウトプット
        grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
        gradient_function = K.function([self.model.input[0],self.model.input[1]], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数
        
        output, grads_val = gradient_function([X, target])
        output, grads_val = output[0], grads_val[0]

        # 重みを平均化して、レイヤーのアウトプットに乗じる
        weights = np.mean(grads_val, axis=(0, 1))
        if self.top50 == True:
            label = np.argsort(weights)
            cam = np.dot(output[:,:,label[-50:]], weights[label[-50:]])
        else:
            cam = np.dot(output, weights)

        # ヒートマップにして合成
        cam = cv2.resize(cam, (self.x.shape[1], self.x.shape[0]), cv2.INTER_LINEAR)
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        
        jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
        jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
        jetcam = (np.float32(jetcam) + self.x * 255 / 2)   # もとの画像に合成

        return jetcam, weights, time.time()-before
