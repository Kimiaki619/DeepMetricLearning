import torch
from torchvision import models
from torch import nn

from sklearn import KMeans
import numpy as np
import pandas as pd
import sys
import cv2
import os
import shutil

DATA_DIR = '../data/'
VIDEOS_DIR = '../data/video/'                        # The place to put the video
TARGET_IMAGES_DIR = '../data/images/target/'         # The place to put the images which you want to execute clustering
CLUSTERED_IMAGES_DIR = '../data/images/clustered/'   # The place to put the images which are clustered
IMAGE_LABEL_FILE ='image_label.csv'                  # Image name and its label

class Image_Clustering:
    def __init__(self,n_clusters=50, model_file=""):
        self.n_clusters = n_clusters
        self.model_file = model_file
    
    def main(self):
        pass

    def model_create(self):
        if self.model_file == "":
            model = models.vgg19(pretrained=True)
            layer = list(model.classifier.children())[:-2]
            model.classifier = nn.Sequential(*layer)        