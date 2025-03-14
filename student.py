import cv2
import os
import pandas as pd

class Student:
    def __init__(self, student_id, name, swipes, dining_dollars):
        self.student_id = student_id 
        self.name = name  
        self.swipes = swipes 
        self.dining_dollars = dining_dollars  
        self.photo = self.load_photo()  

    def load_photo(self):
        photo_path = f"photos/User.{self.student_id}.jpg"
        if os.path.exists(photo_path):
            photo = cv2.imread(photo_path)
            if photo is None:
                print(f"Error: Could not read image file for student {self.student_id}.")
            return photo
        else:
            print(f"Warning: No photo found for student {self.student_id}.")
            return None

    def checkin(self):
        if (self.swipes > 0):
            self.swipes -= 1
            print(f"Using swipes to check in {self.name}, reamining swipes = {self.swipes}")
        elif (self.dining_dollars >= 10):
            self.dining_dollars -= 10.0
            print(f"Using dinning dollars to check in {self.name}, reamining balance = {self.dining_dollars}")
        else:
            print(f"Insufficent dinning dollars and no swipes! cannot checkin {self.name}")
            return

        self.update_csv("data.csv")


    def update_csv(self, csv_file):
        try:
            df = pd.read_csv(csv_file)

            #Find the row where the student's ID matches
            mask = df["ID"] == self.student_id
            if mask.any():
                df.loc[mask, "Swipes"] = self.swipes
                df.loc[mask, "Dining Dollars"] = self.dining_dollars

                #Save the updated CSV file
                df.to_csv(csv_file, index=False)

            else:
                print(f"Error: Student ID {self.student_id} not found in {csv_file}")

        except Exception as e:
            print(f"Error updating CSV file: {e}")