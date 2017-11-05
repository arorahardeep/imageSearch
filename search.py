#!/usr/bin/env python

"""
This program is a helper that loads all the images that need to be searched
"""


from keras.applications import inception_v3, resnet50
from keras.preprocessing import image
from keras.models import Model

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.spatial import distance

import numpy as np
import os
import glob
from os import walk

class ImgSearch:

    _model = None
    _search_path = './data2'
    _img_cols_px = 224
    _img_rows_px = 224
    _img_layers  = 3
    _img_features = None
    _search_scores = None
    _file_names = None
    _match_count = 0

    def __init__(self):
        #self._load_inception3()
        #self._load_resnet50()
        pass

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

    def image_to_vector(self, image_name):
        img_path = os.path.join(self._search_path,image_name)
        img = image.load_img(img_path, target_size=( self._img_cols_px, self._img_rows_px))
        img = image.img_to_array(img)
        img = img.reshape(1, self._img_cols_px, self._img_rows_px, self._img_layers)
        preds = self._model.predict(img)
        return preds

    def create_img_vectors(self, location):
        f = []
        file_names = []
        for( dirpath, dirnames, filenames ) in walk(self._search_path):
            f.extend(filenames)
            for file in f:
                if file.endswith(".jpg"):
                    file_names.append(file.replace(".npy",""))
            break

        file_names = np.asarray(file_names)
        np.save(location + ".label", file_names)

        features = []
        for file_name in file_names:
            print(file_name)
            img_vec = self.image_to_vector(file_name)
            img_vecs = np.array(img_vec)
            features.append(img_vecs)

        features = np.asarray(features)
        np.save(location + ".npy", features)


    def show_image(self, name, label):
        img_path = os.path.join(self._search_path, name)
        img= mpimg.imread(img_path)
        plt.imshow(img)
        plt.title(label)
        plt.suptitle(name)
        plt.show()

    def load_imgs_vectors(self, location):
        if self._file_names is None:
            self._file_names = np.load(location + ".label.npy")
            self._img_features = np.load(location + ".npy")
        #print(self._file_names)

    def search_img(self, comparison_image):
        search_scores = []
        for feat in self._img_features :
            score = np.linalg.norm(feat - comparison_image)
            #score = 1 - distance.cosine(comparison_image, feat)
            search_scores.append(score)

        lowest = sorted(search_scores, key=float, reverse=False)
        self._search_scores = search_scores
        return lowest

    def increment_count(self,base_class, match_class):
        if self.get_class(base_class) == self.get_class(match_class):
            self._match_count = self._match_count + 1

    def show_img_with_score(self, score, base, flist):
        search_index = self._search_scores.index(score)
        fname = self._file_names[search_index]
        if fname not in flist and fname != base:
            flist.append(self._file_names[search_index])
            self.show_image(self._file_names[search_index], "Similarity Score : " + str(score))
            self.increment_count(base, self._file_names[search_index])
            print("Search Score = " + str(score))
        return flist


    def load_img_from_vec(self, image_name):
        fname =  self._file_names.tolist()
        indx = fname.index(image_name)
        return self._img_features[indx], indx

    def get_class(self, carname):
        return carname.split("_0")[0]

def imgs_to_npy():
    query = ImgSearch()
    query.create_img_vectors('./data2/cat_resnet50')

def img_search_frcnn(search_img_name):
    #search_img_name = "Cat_008.jpg"
    query = ImgSearch()
    query.load_imgs_vectors('./data2/reg_cat_2')

    #search_img = query.image_to_vector(search_img_name)
    search_img,indx = query.load_img_from_vec(search_img_name)
    query.show_image(search_img_name, "Base Image")

    search_scores = query.search_img(search_img)
    flist= []
    res = 1
    while (1):
        flist=query.show_img_with_score(search_scores[res],search_img_name, flist)
        if len(flist) >= 5:
            break
        res=res+1


def img_search_normal(search_img_name):
    #search_img_name = "Cat_008.jpg"
    query = ImgSearch()
    query.load_imgs_vectors('./data2/cat_resnet50')

    #search_img = query.image_to_vector(search_img_name)
    search_img,indx = query.load_img_from_vec(search_img_name)
    query.show_image(search_img_name, "Base Image")

    search_scores = query.search_img(search_img)
    flist= []
    res = 1
    while (1):
        flist=query.show_img_with_score(search_scores[res],search_img_name, flist)
        if len(flist) >= 5:
            break
        res=res+1


def all_image_search():
    path = "./data2"
    score = 0
    img_no = 1
    for( dirpath, dirnames, filenames ) in walk(path):
        for f in filenames:
            if f.endswith(".jpg"):
                print("Searching " + f)
                img_no = img_no + 1
                score = score + img_search_new(f)
                print(score)

    print("Total Search = %d, max hit = %d, percent correct = %2.2f"%(img_no, img_no*5, (score/(img_no*5))*100) )

def img_search_new(search_img_name):
    query = ImgSearch()
    #query.load_imgs_vectors('./data2/reg_cat_1')
    query.load_imgs_vectors('./data2/cat_resnet50')

    search_img,indx = query.load_img_from_vec(search_img_name)
    query.show_image(search_img_name, "Base Image")

    search_scores = query.search_img(search_img)
    flist= []
    res = 1
    while (1):
        flist = query.show_img_with_score(search_scores[res],search_img_name, flist)
        if len(flist) >= 5:
            break
        res = res + 1
        print(flist)


    #print(query._match_count)
    return query._match_count

def main():
    #img_search_old()
    #img_search_new()
    #imgs_to_npy()
    #all_image_search()
    #img_search_new("Cat_008.jpg")
    pass

if __name__ == "__main__":
    main()

