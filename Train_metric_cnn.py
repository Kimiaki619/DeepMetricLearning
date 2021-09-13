from keras.applications import vgg19
from keras.engine.topology import Layer
from keras import backend as K
from keras.layers import Input, Activation
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

import ArcFaceLayer

class train_metric_cnn():
    def __init__(self,X_train, Y_train, X_test, Y_test,PATH,MODEL_NAME,MODEL_WEIGHTS,classes=2):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.classes = classes
        self.path = PATH
        self.MODEL_NAME = MODEL_NAME
        self.MODEL_WEIGHTS = MODEL_WEIGHTS

    def train_mobileV2(self, epoch, alpha_=0.5):
        mobile = vgg19.VGG19(include_top=True,input_shape=(224,224,3),weights='imagenet')
        
        # 最終層削除
        mobile.layers.pop()
        v2 = Model(inputs=mobile.input,outputs=mobile.layers[-1].output)
        

        model = self.build_arcface(v2)

        datagen = ImageDataGenerator(rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True)
        
        datagen.fit(self.X_train)

        #cnnの学習
        hist = model.fit_generator(datagen.flow([self.X_train, self.Y_train], self.Y_train, batch_size=32),
                                steps_per_epoch=self.X_train.shape[0] /32,
                                validation_data=([self.X_test, self.Y_test], self.Y_test),
                                epochs=epoch, 
                                verbose=1)

        plt.figure()               
        plt.plot(hist.history['val_acc'],label="val_acc")
        plt.legend(loc="lower right")
        plt.show()
        
        #Modelを保存する
        model_json = model.to_json()
        open(self.path+ self.MODEL_NAME, 'w').write(model_json)
        #Parameterの保存&読み込み
        model.save_weights(self.path + self.MODEL_WEIGHTS)

        return model

    # mobilenetV2と接合して学習
    def build_arcface(self, base_model):
        #add new layers 
        hidden = base_model.output
        yinput = Input(shape=(self.classes,)) #ArcFaceで使用
        # stock hidden model
        c = ArcFaceLayer.Arcfacelayer(self.classes, 30, 0.05)([hidden,yinput]) #outputをクラス数と同じ数に
        prediction = Activation('softmax')(c)
        model = Model(inputs=[base_model.input, yinput], outputs=prediction)

        model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=0.0001, amsgrad=True),
                    metrics=['accuracy'])

        return model
