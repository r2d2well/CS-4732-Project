import cv2
import os
import numpy as np

#Path to dataset
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#Function to load stored face images
def load_faces():
    faces_db = {}
    for file in os.listdir(dataset_path):
        if file.endswith(".jpg"):
            user_id = file.split(".")[1]
            img = cv2.imread(os.path.join(dataset_path, file), cv2.IMREAD_GRAYSCALE)
            faces_db[user_id] = img
    return faces_db

#Function to compare faces
def compare_faces(face, faces_db):
    best_match = None
    best_score = float("-inf")  #Higher = closer match

    for user_id, stored_face in faces_db.items():
        resized_face = cv2.resize(face, (stored_face.shape[1], stored_face.shape[0]))

        result = cv2.matchTemplate(stored_face, resized_face, cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > 0.7:  #Threshold for similarity
            best_match = user_id
            best_score = max_val

    return best_match if best_match else "Unknown"

#Function to register a new user
def add_new_user(face):
    user_id = len(os.listdir(dataset_path)) + 1  #Assign new user ID (Should be student ID in actual deployment)
    file_path = f"{dataset_path}/User.{user_id}.jpg"
    cv2.imwrite(file_path, face)
    print(f"User {user_id} registered successfully!")

#Start the webcam
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Load stored faces
    faces_db = load_faces()

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        user = compare_faces(face, faces_db)

        #Display the results
        text = f"User {user}" if user != "Unknown" else "Unknown (Press 'N' to Register)"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #Register a new face if user presses 'N'
        if user == "Unknown" and cv2.waitKey(1) & 0xFF == ord("n"):
            add_new_user(face)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  #Quits on 'q' 
        break

cam.release()
cv2.destroyAllWindows()