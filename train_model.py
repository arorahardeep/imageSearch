#!/usr/bin/env python

"""
Train the model before search, so that the image vectors are more better
"""

from keras.applications import inception_v3, resnet50
from keras.preprocessing import image
from keras.layers import Dense,Dropout
from keras.models import Model

class TrainSearch:
    _model = None
    _search_path = './data2'
    _img_cols_px = 224
    _img_rows_px = 224
    _img_layers  = 3
    _num_classes = 7
    _class_label = ['Cat','Cat_n_Dog','Cat_n_Man','Cat_n_Women','Dog','Dog_n_Man','Dog_n_Women']

    def __init__(self):
        #self._load_inception3()
        #self._load_resnet50()
        pass

    @staticmethod
    def build_inception(self):
        input_shape = (1, self._img_cols_px, self._img_rows_px, self._img_layers)
        #Input = tf.contrib.keras.layers.Input(shape=input_shape)
        cnn = inception_v3.InceptionV3(weights='imagenet',
                        input_shape = input_shape,
                        include_top=False,
                        pooling='avg')
        for layer in cnn.layers:
            layer.trainable = False
        cnn.trainable = False
        x = cnn.output
        x = Dropout(0.6)(x)
        x = Dense(1024, activation='relu', name='dense01')(x)
        #x = Dropout(0.2)(x)
        x = Dense(512, activation='relu', name='dense02')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu', name='dense03')(x)
        x = Dense(64, activation='relu', name='dense04')(x)
        Output = Dense(self._num_classes, activation='softmax', name='output')(x)
        self._model = Model(cnn.input, Output)

    def _load_resnet50(self):
        base_model = resnet50.ResNet50(weights='imagenet', include_top=True)
        cnn = inception_v3.InceptionV3(weights='imagenet',
                        input_shape = input_shape,
                        include_top=False,
                        pooling='avg')
        for layer in cnn.layers:
            layer.trainable = False
        cnn.trainable = False
        x = cnn.output
        x = Dropout(0.6)(x)
        x = Dense(1024, activation='relu', name='dense01')(x)
        #x = Dropout(0.2)(x)
        x = Dense(512, activation='relu', name='dense02')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu', name='dense03')(x)
        x = Dense(64, activation='relu', name='dense04')(x)
        Output = Dense(self._num_classes, activation='softmax', name='output')(x)
        self._model = Model(cnn.input, Output)