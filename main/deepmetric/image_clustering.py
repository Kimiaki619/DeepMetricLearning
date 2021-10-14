from matplotlib import image
import torch
from torchvision import models
from torch import nn
import torchvision
import torchvision.transforms as transforms

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import sys
import cv2
import os
from progressbar import ProgressBar 
import shutil

#田久保が書いたファイル
import CnnNet

DATA_DIR = '../data/'
VIDEOS_DIR = '../data/video/'                        # The place to put the video
TARGET_IMAGES_DIR = '../data/images/target/'         # The place to put the images which you want to execute clustering
CLUSTERED_IMAGES_DIR = '../data/images/clustered/'   # The place to put the images which are clustered
IMAGE_LABEL_FILE ='image_label.csv'                  # Image name and its label

class Image_Clustering:
    def __init__(self,n_clusters=50, model_file=""):
        self.n_clusters = n_clusters
        self.model_file = model_file
        #gpuでのトレーニング
        #なかったらcpuに切り替わる
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    def main(self):
        pass

    def model_create(self):
        if self.model_file == "":
            model = models.vgg19(pretrained=True)
            layer = list(model.classifier.children())[:-2]
            model.classifier = nn.Sequential(*layer)
            model.to(self.device)
        else:
            #保存したモデルをロードする。
            #自分で学習したモデルをここで呼ぶ
            model = CnnNet.Net()
            model.load_state_dict(torch.load(self.model_file))
            model.to(self.device)
        return model

    def label_images(self):
        print("label imagea ・・・")
        #load model
        model = self.model_create()

        #get images
        images = [f for f in os.listdir(TARGET_IMAGES_DIR) if f[-4:] in ['.png', '.jpg']]
        assert(len(images)>0)
        X = []
        pd = ProgressBar(max_value=len(images))
        for i in range(len(images)):
            feat = self.__featu()

    def __feature_extraction(self, model, img_path):
        #訓練時と推論時で振る舞いが異なるレイヤーがある場合はeval()を実行しないと正しい結果にならない
        model.eval()
        #正規化するためにかく
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        image_set = torchvision.datasets.ImageFolder(img_path,transform)
        
        


model = Image_Clustering(n_clusters=2,model_file="/Users/takubokouakira/Desktop/pytorch_metric/自作CNN/main/deepmetric/models/model.pth").model_create()
print(model)

model_vgg = Image_Clustering(n_clusters=2,model_file="").model_create()
print("modelのvggは")
print(model_vgg.classifier)