from fastapi import FastAPI, UploadFile, HTTPException
import numpy as np
import cv2
from facial_recognition import add_p, delete_p, recognize_p, list_p

app = FastAPI()


@app.post("/add_person")
async def add_person(person_name: str, file: UploadFile):
    ''' function takes a person_name string and an UploadFile as parameters.
         It uses cv2 to decode the file into an image and passes it along with the person_name to the add_p() 
         function from the facial_recognition module. It then returns a message confirming that the person was added to the database.
    '''
    face_image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), -1)
    recognized = add_p(person_name, face_image)
    return {"message": f"Person {person_name} added to the database"} if not recognized else {"message": f" this Person was added before to the database"}


@app.delete("/delete_person/{person_name}")
def delete_person(person_name: str):
    ''' This function takes a person_name string as a parameter and passes it to 
        the delete_p() function from the facial_recognition module. 
        If the person is found in the database, it returns a message confirming that they were deleted;
         otherwise, it raises an HTTPException with status code 400 and an error message. 
    '''

    # check and delete the given person name from datbase
    person_found = delete_p(person_name)

    if person_found:
        return {"message": f"Person {person_name} deleted from the database"}
    else:
        raise HTTPException(
            status_code=400, detail=f"Person {person_name} not found in the database")


@app.post("/recognize_person/")
async def recognize_person(file: UploadFile, distance_metric: str = "cosin", threshold: float = None):
    ''' This function takes an UploadFile and two optional parameters: distance metric (defaults to "cosin") and threshold (defaults to None). 
            It uses cv2 to decode the file into an image and passes it along with distance metric and threshold to recognize_p() from facial_recognition module.
            It then returns either the recognized person or "No matching person found". 
    '''
    face_image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), -1)
    recognized_person = recognize_p(face_image, distance_metric, threshold)
    return {"recognized_person": recognized_person if recognized_person else "No matching person found"}


@app.get("/list_persons/")
async def list_persons():
    '''This finction retreves all the stored users names in the database'''
    return {"persons": list_p()}
