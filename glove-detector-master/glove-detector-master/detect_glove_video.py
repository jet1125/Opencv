# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import argparse
import cv2
import os


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	#blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	layerOutputs = faceNet.forward([])

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []
	boxes = []
	confidences = []
	classIDs = []
	hands = []
	locs = []
	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			print(scores)
			classID = np.argmax(scores)
			print(classID)
			confidence = scores[classID]

			if confidence >0.5:
				box =detection[0:4] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
				hand = frame[startY:endY, startX:endX]
				hand = cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)
				hand = cv2.resize(hand, (224, 224))
				hand = img_to_array(hand)
				hand = preprocess_input(hand)

				#boxes.append([x, y, int(width), int(height)])
				#confidences.append(float(confidence))
				#classIDs.append(classID)
				hands.append(hand)
				locs.append((startX, startY, endX, endY))

	#idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
	labels = ["glove"]
	results = []
	
	if len(faces) > 0:
		hands = np.array(hands, dtype="float32")
		preds = maskNet.predict(hands, batch_size=32)
		"""for i in idxs.flatten():
			x, y = (boxes[i][0], boxes[i][1])
			w, h = (boxes[i][2], boxes[i][3])
			id = classIDs[i]
			confidence = confidences[i]
			if confidence> 0.5:
				results.append((id, labels[id], confidence, x, y, w, h))
				hand = frame[x, y, x+w, y+h]
				hand = cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)
				hand = cv2.resize(hand, (224, 224))
				hand = img_to_array(hand)
				hand = preprocess_input(hand)
				
				hands.append(hand)
				locs.append((x, y, x + w, y + h))

	if len(hands) > 0:
		hands = np.array(hands, dtype="float32")
		preds = maskNet.predict(hands, batch_size = 32)"""
	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# construct the argument parser and parse the arguments

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(["face_detector/", "cross-hands-yolov4-tiny.cfg"])
weightsPath = os.path.sep.join(["face_detector/", "cross-hands-yolov4-tiny.weights"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model("glove_detector.model")
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	locs, preds = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

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

	
				
	#how the output frame
			
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()