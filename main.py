import cv2
import os
import time
import numpy as np
import pandas as pd
import threading
from student import Student

#Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#Global variables for threading
frame = None
running = True
scanning = False
message = ""
user = None
students = []
detected_faces = []

#Load student data
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
    return students

#Face comparison function
def compare_faces(face, students):
    best_match = None
    best_score = float("-inf")  #Higher = closer match

    for student in students:
        try:
            student_photo_gray = cv2.cvtColor(student.photo, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(face, (student_photo_gray.shape[1], student_photo_gray.shape[0]))
            result = cv2.matchTemplate(student_photo_gray, resized_face, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > 0.7:
                best_match = student
                best_score = max_val
        except Exception as e:
            print(f"Error comparing faces: {e}")

    return best_match if best_match else None

#Camera capture function (runs in a separate thread)
def capture_camera():
    global frame, running
    cam = cv2.VideoCapture(0)

    while running:
        ret, temp_frame = cam.read()
        if ret:
            frame = temp_frame

    cam.release()

#Face detection function (runs in a separate thread)
def detect_faces():
    global frame, user, message, detected_faces, scanning

    while running:
        if frame is None:
            continue 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detected_faces = faces

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            user = compare_faces(face, students)

        #If scanning mode is active, attempt to check in
        if scanning and user:
            output = user.checkin()
            time.sleep(2)
            scanning = False 

# Start multi-threading
students = load_all_students()
camera_thread = threading.Thread(target=capture_camera, daemon=True)
face_thread = threading.Thread(target=detect_faces, daemon=True)
camera_thread.start()
face_thread.start()

#Main UI Loop (Runs in the main thread)
print("Starting Camera")

while True:
    if frame is not None:
        display_frame = frame.copy()

        #Draw bounding boxes around detected faces
        for (x, y, w, h) in detected_faces:
            color = (0, 255, 0) if user else (0, 0, 255)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

            #Display name or prompt message above the bounding box
            text = user.name if user else "Unknown"
            cv2.putText(display_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if scanning:
            message = "Scanning"

        else:
            message = ""

        #Display central message
        if message:
            (text_width, text_height), baseline = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            x = (display_frame.shape[1] - text_width) // 2

            cv2.putText(display_frame, message, (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        running = False
        break
    elif key == ord("e"):
        scanning = True

cv2.destroyAllWindows()
