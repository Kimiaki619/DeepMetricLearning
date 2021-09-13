from keras.models import Model
from keras.preprocessing.image import array_to_img

from sklearn.cluster import KMeans
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import Grad_CAM
import time

class Faster_Grad_CAM():
    def __init__(self,model):
        self.model = model

    # compare grad and faster-grad
    def show_result(self, model_small, layer_name, data, no, kmeans, channel_weight, channel_adress, top50=False):
        original, result_grad, result_faster, time0, time1 = [], [], [], [], []
        for i in range(5):
            original.append(data[no[i]]) 
            img0, _, time_0 = Grad_CAM.gradcam(self.model, data[no[i]], layer_name, 1, top50)
            img1, time_1 = self.predict_faster_gradcam(data[no[i]], model_small, kmeans, channel_weight, channel_adress)
            result_grad.append(img0)
            result_faster.append(img1)
            time0.append(time_0)
            time1.append(time_1)

        plt.figure(figsize=(15,10))
        for i in range(5):
            plt.subplot(3,5,i+1)
            plt.axis("off")
            if i == 0:
                plt.title("original")
            plt.imshow(original[i])
        for i in range(5):
            plt.subplot(3,5,i+6)
            plt.axis("off")
            if i == 0:
                time_ = int(np.mean(time1)*1000)
                plt.title("Faster-Grad-CAM \n(%d msec)" % time_)
            plt.imshow(array_to_img(result_faster[i]))
        for i in range(5):
            plt.subplot(3,5,i+11)
            plt.axis("off")
            if i == 0:
                time_ = int(np.mean(time0)*1000)
                plt.title("Grad-CAM \n(%d msec)" % time_)
            plt.imshow(array_to_img(result_grad[i]))
        plt.show()

    def train_faster_gradcam(self,x_normal, x_anomaly, clusters=10):
        # Arcfaceを削除
        model_embed = Model(self.model.get_layer(index=0).input, [self.model.layers[-13].get_output_at(-1), model.layers[-4].get_output_at(-1)])

        # pa class data
        _, vector_normal = model_embed.predict(x_normal)

        # gu class data
        _, vector_anomaly = model_embed.predict(x_anomaly)# shape[(len(x), 3, 3, 480), (len(x), 1280)]

        # k-means
        kmeans = KMeans(n_clusters=clusters, random_state=0).fit(vector_anomaly)
        labels = kmeans.labels_

        # channel database
        channel_weight, channel_adress = [], []
        temp_weight = np.zeros((clusters, 480))# 480="block_16_expand_relu".output
        print("Making Database...")
        for i in range(len(labels)):
            # x_anomalyについて一個ずつ重みを加算していく
            _, weight, _ = Grad_CAM.gradcam(model, x_anomaly[i], "block_16_expand_relu", 1, False)
            temp_weight[labels[i]] += weight #要確認
            print(i+1,"/",len(labels))

        for i in range(clusters):
            number = np.where(labels == i, 1, 0) #クラスタ内の個数
            average_weight = temp_weight[i] / np.sum(number) #重みの平均
            weight_adress = np.argsort(average_weight)
            channel_adress.append(weight_adress[-50:])
            channel_weight.append(average_weight[weight_adress[-50:]])

        return model_embed, kmeans, np.array(channel_weight), np.array(channel_adress), vector_normal

    def predict_faster_gradcam(self,x, kmeans, channel_weight, channel_adress):
        before = time.time()
        channel_out, vector = self.model.predict(np.expand_dims(x, axis=0))
        channel_out = channel_out[0]
        cluster_no = kmeans.predict(vector)
        # レイヤーのアウトプットに乗じる
        cam = np.dot(channel_out[:,:,channel_adress[cluster_no][0]], channel_weight[cluster_no][0])

        # ヒートマップにして合成
        cam = cv2.resize(cam, (x.shape[1], x.shape[0]), cv2.INTER_LINEAR)
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        
        jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
        jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
        jetcam = (np.float32(jetcam) + x*255 / 2)   # もとの画像に合成

        return jetcam, time.time()-before
