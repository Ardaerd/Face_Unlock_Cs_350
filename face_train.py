import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    encodes = []
    all_names = []
    train_dir = os.listdir('dataset/')
    print(train_dir)
    for person in train_dir:
        p = os.listdir("dataset/" + person)
        for picture_human in p:
            print("dataset/" + person + "/" + picture_human)
            front = face_recognition.load_image_file("dataset/" + person + "/" + picture_human)
            print(front.shape)
            h, w, _ = front.shape
            face_location = (0, w, h, 0)
            print(w, h)
            face_enc = face_recognition.face_encodings(front, known_face_locations=[face_location])
            face_enc = np.array(face_enc)
            face_enc = face_enc.flatten()
        
            encodes.append(face_enc)
            all_names.append(person)
    print(np.array(encodes).shape)
    print(np.array(all_names).shape)
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(encodes,all_names)

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


if __name__ == "__main__":
    print("Training KNN classifier...")
    classifier = train("dataset", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")    