from PIL import Image
import glob, cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
import keras
from sklearn.model_selection import train_test_split

def load_images(path):
    result_img = []
    file_name = glob.glob(path + "*")
    for name in file_name:
        img = Image.open(name)
        img = image.img_to_array(img)
        img = cv2.resize(img,(224, 224))
        result_img.append(img)
    return np.array(result_img)