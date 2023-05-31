# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import h5py
# import _pickle as cPickle
import pickle
import face_recognition

import os
from PIL import Image, ImageDraw

argp = argparse.ArgumentParser()
argp.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(argp.parse_args())
with open("trained_knn_model.clf", 'rb') as f:
    knn_clf = pickle.load(f)

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    cam = cv2.VideoCapture(0)
else:
    cam = cv2.VideoCapture(args["video"])

while True:
    (grabbed, image1) = cam.read()
    if args.get("video") and not grabbed:
        break
    picture = image1[:, :, ::-1]
    X_face_locations = face_recognition.face_locations(picture)
    if len(X_face_locations) != 0:
        encoding_face = face_recognition.face_encodings(picture, known_face_locations=X_face_locations, num_jitters=1)
        print(np.array(encoding_face).shape)
        closest_distances = knn_clf.kneighbors(encoding_face, n_neighbors=1)
        is_match = [closest_distances[0][i][0] <= 0.4 for i in range(len(X_face_locations))]
        guesses = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                   zip(knn_clf.predict(encoding_face), X_face_locations, is_match)]
        for n, (top, right, bottom, left) in guesses:
            if n not in "unknown":
                os.popen('gnome-screensaver-command -d && xdotool key Return')
            cv2.rectangle(image1, (left, bottom), (right, top), (0, 255, 0), 2)
            cv2.putText(image1, "{}".format(n), (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        os.popen('gnome-screensaver-command -a')
    cv2.imshow("output image", image1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
