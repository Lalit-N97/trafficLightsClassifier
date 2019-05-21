# import os
import numpy as np
from pathlib import Path
import keras.backend as K
from keras.preprocessing import image
# import h5py
import matplotlib.pyplot as plt
import math
# from PIL import Image
# from scipy import ndimage
# import skimage.data

def mean_pred(y_true, y_pred):
	return K.mean(y_pred)

# def load_dataset(data_dir):
#     test_images = np.array([])
#     test_labels = np.array([])
#     train_images = np.array([])
#     train_labels = np.array([])
#     images = []
#     labels = []
#     directories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
#     for idx, d in enumerate(directories):
#         label_dir = os.path.join(data_dir, d)
#         file_names = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".png")]

#         for f in file_names:
#             images.append(skimage.data.imread(f))
#             labels.append(idx)
#         # print (np.array(images).shape)
#         img_num = len(images)
#         # print (img_num)
#         t = math.floor(img_num*0.1)
#         # print(t)
#         test_images = np.append(test_images, images[:t])
#         train_images = np.append(train_images, images[t:])
#         test_labels = np.append(test_labels, labels[:t])
#         train_labels = np.append(train_labels, labels[t:])
#         images = []
#         labels = []
#     print(test_images.shape, train_images.shape)        
#     return train_images, train_labels, test_images, test_labels


def load_dataset(data_dir):
    p = Path(data_dir)
    dirs = p.glob("*")
    images = []
    labels = []
    test_images = []
    test_labels = []
    train_images = []
    train_labels = []
    label_dict = {"green" : 0, "yellow" : 1, "red" : 2, "unknown" : 3}
    for d in dirs:
        label = str(d).split("/")[-1]
        for img_path in d.glob("*.png"):
            img = image.load_img(img_path, target_size=(100,100))
            img_array = image.img_to_array(img)
            images.append(img_array)
            labels.append(label_dict[label])
        img_num = len(images)
        t = math.floor(img_num*0.1)
        test_images += images[:t]
        train_images += images[t:]
        test_labels += labels[:t]
        train_labels += labels[t:]
        images = []
        labels = []
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y