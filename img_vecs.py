#!/usr/bin/env python

"""
This program loads creates image vectors for each image bounding box
@author : Hardeep Arora
@date   : 22 Sep 2017

"""

import pandas as pd
from keras.applications import inception_v3, resnet50
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

class Img2Vecs:

    _model = None
    _search_path = './data2'
    _img_cols_px = 224
    _img_rows_px = 224
    _img_layers  = 3

    def __init__(self):
        #self._load_inception3()
        self._load_resnet50()
        #pass

    def _load_inception3(self):
        base_model = inception_v3.InceptionV3(weights='imagenet', include_top=True)
        #base_model.summary()
        layer_out = base_model.get_layer('avg_pool')
        self._model = Model(inputs=base_model.input, outputs=layer_out.output)

    def _load_resnet50(self):
        base_model = resnet50.ResNet50(weights='imagenet', include_top=True)
        #base_model.summary()
        layer_out = base_model.get_layer('avg_pool')
        self._model = Model(inputs=base_model.input, outputs=layer_out.output)

    def crop_resize_img(self, image_name, x1, y1, x2, y2):
        img = Image.open(os.path.join(self._search_path,image_name))
        img = img.crop((x1,y1,x2,y2))
        img = img.resize((self._img_cols_px, self._img_rows_px),Image.ANTIALIAS)
        return img

    def image_to_vector(self, image_name, x1, y1, x2, y2):
        img = self.crop_resize_img(image_name, x1, y1, x2, y2)
        img = np.array(img)
        #plt.imshow(img)
        #plt.show()
        img = img.reshape(1, self._img_cols_px, self._img_rows_px, self._img_layers)

        preds = self._model.predict(img)
        return image_name, preds

    def img_calc_areas(self, img_region_path):
        imgs_df = pd.read_csv(img_region_path)
        imgs_df['area'] = (imgs_df['x2'] - imgs_df['x1']) * (imgs_df['y2'] - imgs_df['y1'])
        imgs_df.loc[imgs_df.groupby(['image_name'])['area'].idxmax()].to_csv("unq_img_reg.csv",index=False)

    def img_regions_2_vectors(self, img_region_path, npy_file):
        """
        Loads the image file
        :param path: The path of the file containing image name and its bounding boxes
        :return:
        """
        imgs_df = pd.read_csv(img_region_path)
        features = []
        file_names = []
        for i in range(0,imgs_df.shape[0]):
            img_nm, vec = self.image_to_vector(imgs_df['image_name'].iloc[i],
                                               imgs_df['x1'].iloc[i],
                                               imgs_df['y1'].iloc[i],
                                               imgs_df['x2'].iloc[i],
                                               imgs_df['y2'].iloc[i])
            file_names.append(img_nm)
            features.append(vec)

        file_names = np.asarray(file_names)
        np.save(npy_file + ".label", file_names)

        features = np.asarray(features)
        np.save(npy_file + ".npy", features)

def main():
    iv = Img2Vecs()
    iv.img_regions_2_vectors("./data2/img_regions_cats_1.csv","./data2/reg_cat_2")
    #iv.img_calc_areas("./data/img_regions.csv")

if __name__ == "__main__":
    main()
