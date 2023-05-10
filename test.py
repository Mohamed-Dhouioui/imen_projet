from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
from efficientnet import EfficientNetB5
# import efficientnet.tfkeras
from keras.preprocessing.image import img_to_array
# from keras.models import load_model
from keras.models import load_model
import numpy as np
import imutils
import os, time
import datetime
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import sys
from utils import label_map_util
from utils import visualization_utils as vis_util
import pickle
import skimage
from skimage import io
import json


image_path = './Image'
MODEL_PATH = "./models/image_classification"

path = './Image'
# Initialize dlib's face detector (HOG-based) and then create
# The facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/object_detection/shape_predictor_68_face_landmarks.dat")
# Image info
img_file = './Image/teste.jpg'
img_name = 'teste.jpg'
frame = cv2.imread(os.path.join(image_path, img_name))
frame1 = frame
image = imutils.resize(frame, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)

# Loop over the face detections
for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        clone = image.copy()
        cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        for (x, y) in shape[i:j]:
            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        roi = image[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        if (name == "right_eye" or name == "left_eye" or name == "nose" or name == "mouth"):
            cv2.imwrite(os.path.join(path, name + ".jpg"), roi)

# Custom model for under eye object detection trained using tensorflow implementation of faster-rcnn based on inception-v2

PATH_TO_CKPT = "./object_detection/inference_graph/frozen_inference_graph.pb"

# Path to label map file
PATH_TO_LABELS = "./object_detection/training/labelmap.pbtxt"

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image = cv2.imread(os.path.join(image_path, img_name))
image_expanded = np.expand_dims(image, axis=0)
image_height, image_width, _ = image.shape
# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Draw the results of the detection (aka 'visulaize the results')

confidence_threshold = 0.6

# Find indices of eye detections with confidence above the threshold
eye_indices = np.where((classes[0] == 1) & (scores[0] > confidence_threshold))[0]
i = 1
for index in eye_indices:
    ymin, xmin, ymax, xmax = boxes[0][index]
    ymin = int(ymin * image_height)
    xmin = int(xmin * image_width)
    ymax = int(ymax * image_height)
    xmax = int(xmax * image_width)
    # Crop the eye region from the original image using the calculated bounding box coordinates
    under_eye = image[ymin:ymax, xmin:xmax]
    cropped = imutils.resize(under_eye, width=250, inter=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(path, "undereye"+str(i) + ".jpg"), cropped)
    i = int(i)
    i +=1
print("[INFO] loading model...")
model_eye = load_model(os.path.join(MODEL_PATH, "Eyes_4(75).h5"))
graph = tf.get_default_graph()
frame = cv2.imread(os.path.join(path, "right_eye.jpg"))
right_eye_img = "righteye" + img_name
cv2.imwrite(os.path.join("./static", right_eye_img), frame)
frame = imutils.resize(frame, width=400)
# prepare the image to be classified by our deep learning network
image = cv2.resize(frame, (75, 75))
image = np.float32(image / 255)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
with graph.as_default():
    right_eye = model_eye.predict(image)[0][0]

frame = cv2.imread(os.path.join(path, "left_eye.jpg"))
left_eye_img = "lefteye" + img_name
cv2.imwrite(os.path.join("./static", left_eye_img), frame)
frame = imutils.resize(frame, width=400)
# prepare the image to be classified by our deep learning network
image = cv2.resize(frame, (75, 75))
image = np.float32(image / 255)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

with graph.as_default():
    left_eye = model_eye.predict(image)[0][0]

del model_eye, graph

print("[INFO] loading model...")
model_nose = load_model(os.path.join(MODEL_PATH, "Nose(50).h5"))
graph = tf.get_default_graph()
frame = cv2.imread(os.path.join(path, "nose.jpg"))
nose_img = "nose" + img_name
cv2.imwrite(os.path.join("./static", nose_img), frame)
frame = imutils.resize(frame, width=400)
# Prepare the image to be classified by our deep learning network
image = cv2.resize(frame, (50, 50))
image = np.float32(image / 255)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

with graph.as_default():
    nose = model_nose.predict(image)[0][0]

del model_nose,graph

print("[INFO] loading model...")
model_mouth = load_model(os.path.join(MODEL_PATH, "Mouth(50).h5"))
graph = tf.get_default_graph()
frame = cv2.imread(os.path.join(path, "mouth.jpg"))
mouth_img = "mouth" + img_name
cv2.imwrite(os.path.join("./static", mouth_img), frame)
frame = imutils.resize(frame, width=400)
# prepare the image to be classified by our deep learning network
image = cv2.resize(frame, (50, 50))
image = np.float32(image / 255)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

with graph.as_default():
    mouth = model_mouth.predict(image)[0][0]

del model_mouth,graph

print("[INFO] loading model...")
model_undereye = load_model(os.path.join(MODEL_PATH, "Undereyes_4.h5"))
graph = tf.get_default_graph()
frame = cv2.imread(os.path.join(path, "undereye1.jpg"))
left_undereye_img = "leftundereye" + img_name
cv2.imwrite(os.path.join("./static", left_undereye_img), frame)
frame = imutils.resize(frame, width=400)
# prepare the image to be classified by our deep learning network
image = cv2.resize(frame, (50, 50))
image = np.float32(image / 255)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

with graph.as_default():
    undereye1 = model_undereye.predict(image)[0][0]

frame = cv2.imread(os.path.join(path, "undereye2.jpg"))
right_undereye_img = "rightundereye" + img_name
cv2.imwrite(os.path.join("./static", right_undereye_img), frame)
frame = imutils.resize(frame, width=400)
# prepare the image to be classified by our deep learning network
image = cv2.resize(frame, (50, 50))
image = np.float32(image / 255)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

with graph.as_default():
    undereye2 = model_undereye.predict(image)[0][0]

del model_undereye,graph

print("[INFO] loading model...")
model_face = load_model(os.path.join(MODEL_PATH, "Faces(100).h5"))
graph = tf.get_default_graph()
frame = cv2.imread(os.path.join(path, img_name))
skin_img = "skin" + img_name
cv2.imwrite(os.path.join("./static", skin_img), frame)
frame = imutils.resize(frame, width=400)
# prepare the image to be classified by our deep learning network
image = cv2.resize(frame, (100, 100))
image = np.float32(image / 255)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

with graph.as_default():
    face = model_face.predict(image)[0][0]

del model_face,graph

print("[INFO] calculating prediction...")

final_prediction = (((left_eye + right_eye) / 2) * 0.4) + (((undereye1 + undereye2) / 2) * 0.55) + (
            ((nose + mouth + face) / 3) * 0.05)

label = 'Final Prediction'
label = "{}: {:.5f}%".format(label, final_prediction * 100)
frame_final = cv2.putText(frame1, label, (10, 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imwrite(os.path.join("./static", img_name), frame_final)

os.remove(os.path.join(image_path, img_name))

print("Right Eye: " + str(right_eye) + "\n" + "Left Eye: " + str(left_eye) + "\n"
      + "Nose: " + str(nose) + "\n" + "Mouth: " + str(mouth) + "\n" + "Undereye Left: " + str(undereye1)
      + "\n" + "Undereye Right: " + str(undereye2) + "\n" + "Face: " + str(
    face) + "\n" + "Final Prediction: " + str(final_prediction))