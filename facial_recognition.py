from deepface import DeepFace
from deepface.commons import distance
from fastapi import HTTPException

import cv2
import os

models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]


def compare_features(features, db_features, distance_metric='cosine'):

    if distance_metric == 'cosine':
        dist = distance.findCosineDistance(features, db_features)
    elif distance_metric == 'euclidean':
        dist = distance.findEuclideanDistance(features, db_features)
    elif distance_metric == 'euclidean_l2':
        dist = distance.findEuclideanDistance(distance.l2_normalize(
            features), distance.l2_normalize(db_features))
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)

    return dist


def add_p(person_name, face_image):
    '''
    Add a person's face to the database
    person_name: str, name of the person
    face_image: str, path to the image of the face
    '''

    #check if the database is exsist 
    file_found = False
    try:
        with open("face_db.txt", "r") as f:
            lines = f.readlines()
            file_found =True
    except:
        file_found=False

    recognized = None

    # check if the person is exsist in the database before 
    if file_found:
        recognized = recognize_p(face_image)

    if recognized is None :
        # extract features from the face image
        embedding_objs = DeepFace.represent(face_image, model=models[0])
        features = embedding_objs[0]["embedding"]

                # write the features to the database file
        try:
            with open('face_db.txt', 'a') as f:
                f.write(f'{person_name}: {features}\n')
        except FileNotFoundError:
            raise HTTPException(
                status_code=400, detail="Error while adding the person features ")
        return False

    else:
        return True


def delete_p(person_name):
    '''
    Delete a person from the database
    person_name: str, name of the person
    '''
    # read the database into a list of lines
    try:
        with open("face_db.txt", "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Database not found")

    person_found = False
    # write back the lines without the line for the person to delete
    with open('face_db.txt', 'w') as f:
        for line in lines:
            if not line.startswith(person_name):
                f.write(line)
            else:
                person_found = True
    return person_found


def recognize_p(face_image, distance_metric='cosin', threshold=None):
    '''
    Recognize a person in the image
    face_image: str, path to the image of the face
    distance_metric:  str, the way to calculate the distnce between 2 embeddings 
    threshold:  float, the maximum value to determine if the person is verfied or not based on calculated distance between embeddings
    '''
    # extract features from the face image
    embedding_objs = DeepFace.represent(face_image, model=models[0])
    features = embedding_objs[0]["embedding"]

    # read the database into a list of lines
    try:
        with open("face_db.txt", "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Database not found")

    # compare the features of the face image with the features in the database
    min_distance = float('inf')
    recognized_person = None
    for line in lines:
        line = line.strip().split(': ')
        person_name = line[0]
        db_features = eval(line[1])
        dist = compare_features(features, db_features)

        # get the value of threshold based on the model and distance_metric if it's not given by the user
        if threshold is None:
            threshold = distance.findThreshold(models[0], distance_metric)

        if dist < min_distance and dist < threshold:
            min_distance = dist
            recognized_person = person_name

    # return the name of the recognized person
    return recognized_person


def list_p():
    '''
    List all the persons in the database
    '''
    # read the database into a list of lines
    try:
        with open("face_db.txt", "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Database file not found")

    # return the names of all the persons
    return [line.strip().split(': ')[0] for line in lines]
