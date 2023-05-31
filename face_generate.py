# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import h5py
import _pickle as cPickle

import os

argp = argparse.ArgumentParser()
argp.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(argp.parse_args())

numb = 0;
f_count = 0
detect = dlib.get_frontal_face_detector()
print("enter the person name")
name = input()
name_file = "dataset/" + name

if os.path.exists(name_file):
    print("Folder exist")
else:
    os.mkdir(name_file)

if not args.get("video", False):
    cam = cv2.VideoCapture(0)

else:
    cam = cv2.VideoCapture(args["video"])

while True:
    if f_count % 5 == 0:
        print("keyframe")
        (grabbed, picture) = cam.read()

        if args.get("video") and not grabbed:
            break
        picture = imutils.resize(picture, width=500)
        gr = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
        rects = detect(gr, 1)
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cro = picture[y: y + h, x: x + w]
            picture_out = cv2.resize(cro, (108, 108))
            fr = os.path.join(name_file + "/", str(numb) + "." + "jpg")
            numb = numb + 1
            cv2.imwrite(fr, picture_out)
            cv2.rectangle(picture, (x, y), (x + w, y + h), (0, 255, 0), 2)
        f_count = f_count + 1
    else:
        f_count = f_count + 1
        print("redundant frame")
    if numb > 51:
        break
    cv2.imshow("output image", picture)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
