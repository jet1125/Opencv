# Python ≥3.5 is required
import sys

#from trainmodel.class1 import ResidualUnit
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Is this notebook running on Colab?
#IS_COLAB = "google.colab" in sys.modules

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

assert tf.__version__ >= "2.0"


# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import partial
#from class1 import ResidualUnit
#import class1 #import DefaultConv2D





imagePaths = list(paths.list_images("D:/KUME/Opencv/trainmodel/ada"))
data = []
labels = []

for imagePath in imagePaths:
	# extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]
    
	# load the input image (224x224) and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
	# update the data and labels lists, respectively
    data.append(image)
    if label == "with_glove":
        labels.append(1)
    elif label == "without_glove":
        labels.append(0)
    
print(label)
data = np.array(data)
labels = np.array(labels)
data_mean = data.mean(axis = 0, keepdims = True)
data_std = data.std(axis = 0, keepdims = True) + 1e-7
data = (data-data_mean) / data_std
data = data[..., np.newaxis]

"""X_train, X_test, y_train, y_test = train_test_split(np.array(data), np.array(labels), 
                            test_size=0.2, random_state=42, stratify=np.array(labels))

#split = int(len(data)* 0.8)
#X_train, X_valid = np.array(data[:split]), np.array(data[split:])
#y_train, y_valid = np.array(labels[:split]), np.array(labels[split:])
X_train, X_valid, y_train, y_valid = train_test_split(np.array(X_train), np.array(y_train), 
                            test_size=0.2, random_state=42, stratify=np.array(y_train))
X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std
X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]
"""


#train_test_split(x_data, y_data, test_size=0.3, random_state=777, stratify=y_data)
DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="SAME", use_bias=False)

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'skip_layers': self.skip_layers,
            'activation': self.activation,
        
        })
        return config

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
        #return self.activation(Z)



model = keras.models.Sequential()
model.add(DefaultConv2D(64, kernel_size=7, strides=2,
                        input_shape=[224, 224, 3]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(2, activation="softmax"))

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

#checkpoint_path = "D:/KUME/Opencv/trainmodel/ckp"
#model = keras.models.load_model("D:/KUME/Opencv/trainmodel/glove_detector_model.h5")
#model = keras.models.load_model("D:/KUME/Opencv/trainmodel/glove_detector_model.h5", 
#                        custom_objects={'ResidualUnit': ResidualUnit, 'DefaultConv2D' : DefaultConv2D})
#skip_layers = []
#model.load_weights(checkpoint_path)
#y_pred = model.predict(X_train[:])
#model = load_model("D:/KUME/Opencv/trainmodel/glove_detector.model",
#                        custom_objects={'ResidualUnit': ResidualUnit, 'DefaultConv2D' : DefaultConv2D})

#with open("glove.json", "r") as fp:
#    model = model_from_json(fp.read(), custom_objects={"ResidualUnit" : ResidualUnit, "DefaultConv2D" : DefaultConv2D})

model.load_weights("glove_weights.h5")
X_new = data[:]
y_pred = model.predict(X_new)

print(y_pred)
print(labels[:])
#print(y_pred)