import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.models import Sequential
from skimage.transform import resize   # for resizing image
from sklearn.externals import joblib
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, InputLayer, Dropout
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight



filename = 'finalized_model.sav'
data = pd.read_csv('mapping.csv')     # reading the csv file

X = [ ]     # creating an empty array
# for img_name in data.Image_ID:
#     img = plt.imread('train_frames/' + img_name)
#     X.append(img)  # storing each image in array X

X = [plt.imread('train_frames/' + img_name) for img_name in data.Image_ID]

X = np.array(X)    # converting list to array

y = data.Class
dummy_y = np_utils.to_categorical(y)

# image = []
# for i in range(0,X.shape[0]):
#     a = resize(X[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
#     image.append(a)
image = [resize(X[i], preserve_range=True, output_shape=(224,224)).astype(int) for i in range(0,X.shape[0])]

X = np.array(image)


X = preprocess_input(X, mode='tf')      # preprocessing the input data


X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)    # preparing the validation set


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    # include_top=False to remove the top layer

X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
X_train.shape, X_valid.shape

X_train = X_train.reshape(208, 7*7*512)      # converting to 1-D
X_valid = X_valid.reshape(90, 7*7*512)

train = X_train/X_train.max()      # centering the data
X_valid = X_valid/X_train.max()


# i. Building the model
model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='relu', input_dim=7*7*512))   # hidden layer
model.add(Dropout(0.5))      # adding dropout
model.add(Dense(units=512, activation='relu'))    # hidden layer
model.add(Dropout(0.5))      # adding dropout
model.add(Dense(units=256, activation='relu'))    # hidden layer
model.add(Dropout(0.5))      # adding dropout
model.add(Dense(3, activation='softmax'))            # output layer

# ii. Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# iii. Training the model
# model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid))

class_weights = compute_class_weight('balanced',np.unique(data.Class), data.Class)  # computing weights of different classes

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]      # model check pointing based on validation loss

model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid), class_weight=class_weights, callbacks=callbacks_list)

