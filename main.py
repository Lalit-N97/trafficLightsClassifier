import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Flatten, ZeroPadding2D
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, Dropout
# from keras.layers import GlobalMaxPolling2D, Maxpooling2D
from keras.models import Model
from kt_utils import *

data_dir = "./data/"
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset(data_dir)


# Normalize image vectors
X_train = X_train_orig/255
X_test = X_test_orig/255
classes = 4
Y_train = convert_to_one_hot(Y_train_orig, classes)
Y_test = convert_to_one_hot(Y_test_orig, classes)
print(X_train.shape , X_test.shape)
print(Y_train.shape, Y_test.shape)