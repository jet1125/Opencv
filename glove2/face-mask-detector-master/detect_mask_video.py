# USAGE
# python detect_mask_video.py

# import the necessary packages
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
from imutils.video import VideoStream
import numpy as np
#import argparse
import imutils
import time
import cv2
import os
from functools import partial

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

model.load_weights("glove_weights.h5")

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1 /  255.0, (416, 416), swapRB=True, crop=False)

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	layerOutputs = faceNet.forward(output_names)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []
	boxes = []
	confidences = []
	classIDs = []
	# loop over the detections
	#print(np.array(layerOutputs).shape)
	for output in layerOutputs:
		for detection in output:

			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > 0.6:
				box = detection[0:4] * np.array([w, h, w, h])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height/2))
				x_bound = x - 20
				y_bound = y - 20
				w_bound = width + 40
				h_bound = height + 40

				boxes.append([x_bound, y_bound, int(w_bound), int(h_bound)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				try:
					face = frame[y_bound:(y_bound + h_bound), x_bound:(x_bound + w_bound)]
					#face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
					
					#print(face.shape())
					face = cv2.resize(face, (224, 224))
					#face = img_to_array(face)
					#face = preprocess_input(face)
					
					#face = img_to_array(face)
					#face = preprocess_input(face)
					#face = np.asarray(face)
					#print(type(face))
					face_mean = face.mean(axis = 0, keepdims = True)
					face_std = face.std(axis = 0, keepdims = True) + 1e-7
					face = (face-face_mean) / face_std
					face = face[..., np.newaxis]
					faces.append(face)
					#print(face)
					locs.append((x_bound, y_bound, x_bound + w_bound, y_bound + h_bound))
				except Exception as e:
					print(str(e))
				


	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

	results = []
	
	if len(idxs) > 0: 
		for i in idxs.flatten():
        	# extract the bounding box coordinates
			ax, ay = (boxes[i][0], boxes[i][1])
			aw, ah = (boxes[i][2], boxes[i][3])
			id = classIDs[i]
			confidence = confidences[i]
			results.append((id, labels[id], confidence, ax, ay, aw, ah))
			faces = np.array(faces, dtype="float32")
			#preds = maskNet.predict(faces, batch_size=32)
			try:
				preds = maskNet.predict(faces, batch_size = 32)
			except:
				print("adaa")


	return (locs, preds)



	"""for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:	
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)
"""
# construct the argument parser and parse the arguments

# load our serialized face detector model from disk
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import partial
DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="SAME", use_bias=False)

output_names = []
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(["face_detector/", "cross-hands.cfg"])
weightsPath = os.path.sep.join(["face_detector/", "cross-hands.weights"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
ln = faceNet.getLayerNames()
labels = ["hand"]
for i in faceNet.getUnconnectedOutLayers():
	output_names.append(ln[i - 1])
# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
#with open("glove_cnn.json", "r") as fp:
#    glove_model = model_from_json(fp.read(), custom_objects={"DefaultConv2D" : DefaultConv2D})

#glove_model.load_weights("glove_cnn_weights.h5")
maskNet = model
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = cv2.VideoCapture(0)
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	rval, frame = vs.read()
	#frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(withoutMask, mask) = pred
		#print(mask, withoutMask)
		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
#vs.stop()