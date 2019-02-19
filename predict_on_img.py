import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import np_utils
from skimage.transform import resize   # for resizing image
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import glob
from keras.models import model_from_json
from sklearn.externals import joblib
#model = joblib.load('finalized_model.sav')


label_dict = {'0': 'neither JERRY nor TOM', 
'1': 'JERRY', 
'2': 'TOM'}


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    # include_top=False to remove the top layer
# Model reconstruction from JSON file
#with open('model_architecture.json', 'r') as f:
#    model = model_from_json(f.read())
model = load_model('weights.best.hdf5')
#model.load_weights('model_weights.h5')
test = pd.read_csv('mapping.csv')
test_frames = glob.glob('train_frames/*.jpg')
test_image = []
for img_name in test.Image_ID:
    img = plt.imread('train_frames/' + img_name)

    test_image.append(img)
test_img = np.array(test_image)

test_image = []
for i in range(0,test_img.shape[0]):
    a = resize(test_img[i], preserve_range=True, output_shape=(224,224)).astype(int)
    test_image.append(a)
test_image = np.array(test_image)

# preprocessing the images
test_image = preprocess_input(test_image, mode='tf')

# extracting features from the images using pretrained model
test_image = base_model.predict(test_image)

# converting the images to 1-D form
test_image = test_image.reshape(298, 7*7*512)

# zero centered images
test_image = test_image/test_image.max()

predictions = model.predict_classes(test_image)
font = cv2.FONT_HERSHEY_SIMPLEX
for ind, img_path in enumerate(test.Image_ID):
	image = cv2.imread('train_frames/' + img_path)
	h, w = image.shape[:2]
	cv2.putText(image,label_dict[str(predictions[ind])],(int(h/2), int(w/2)), font, 4,(255,255,255),2,cv2.LINE_AA)
	print(img_path, str(predictions[ind]))
	# cv2.startWindowThread()
	cv2.namedWindow("preview")
	cv2.imshow("preview", image)
	cv2.waitKey()	
	# print(img_path, predictions[ind])
	# cv2.destroyAllWindows()
