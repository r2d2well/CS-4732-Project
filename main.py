import cv2
import os
import time
import numpy as np
import pandas as pd
from student import Student

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
user = None
scanning = False

#Function to load stored face images
def load_all_students():
    print("Loading Student Data")
    students = []
    csv_filePath = "data.csv"
    try:
        df = pd.read_csv(csv_filePath)
        for _, row in df.iterrows():
            student = Student(
                student_id=row["ID"],
                name=row["Name"],
                swipes=int(row["Swipes"]),
                dining_dollars=float(row["Dining Dollars"]),
            )
            students.append(student)

    except Exception as e:
        print(f"Error loading students from {csv_filePath}: {e}")

    return students  #Returns a list of Student objects

def add_new_user(face):
    user_id = len(os.listdir("./photos")) + 1  # Assign new user ID
    file_path = f"./photos/User.{user_id}.jpg"
    cv2.imwrite(file_path, face)
    print(f"User {user_id} registered successfully!")

#Function to compare faces
def compare_faces(face, students):
    best_match = None
    best_score = float("-inf")  #Higher = closer match

    for student in students:
        student_photo_gray = cv2.cvtColor(student.photo, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(face, (student_photo_gray.shape[1], student_photo_gray.shape[0]))

        result = cv2.matchTemplate(student_photo_gray, resized_face, cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > 0.7:  #Threshold for similarity
            best_match = student
            best_score = max_val

    return best_match if best_match else None

students = load_all_students()
#Start the webcam
print("Starting Camera")
cam = cv2.VideoCapture(0)
message = ""
message_time = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if cv2.waitKey(1) & 0xFF == ord("e"):
        message = "Scanning"
        scanning = True

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        user = compare_faces(face, students)

        #Display the results
        if scanning and user != None:
            output = user.checkin()
            if (output == None):
                text = f"User {user.name} checkedin"
            else:
                text = output
                message_time = time.time()
            scanning = False
            message = ""

        else:
            text = f"User {user.name} (Press E to checkin)" if user != None else "Unknown (press 'n' to register new user)"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if user == None and cv2.waitKey(1) & 0xFF == ord("n"):
            add_new_user(face)

        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)

    frame_width = frame.shape[1]

    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, .7, 2)[0]
    text_x = (frame_width - text_size[0]) // 2

    cv2.putText(frame, message, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  #Quits on 'q' 
        break

cam.release()
cv2.destroyAllWindows()