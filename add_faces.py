import cv2
import pickle
import numpy as np
import os

def collect_face_data():
    names = []  # Initialize an empty list to store names
    faces_data = []

    while len(names) < 100:
        name = input("Enter Your Name: ")
        dept = input("Enter Department: ")
        year = input("Enter Year: ")

        video = cv2.VideoCapture(0)
        facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

        i = 0

        while True:
            ret, frame = video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                crop_img = frame[y:y + h, x:x + w, :]
                resized_img = cv2.resize(crop_img, (50, 50))
                if len(faces_data) < 100 and i % 10 == 0:
                    faces_data.append(resized_img)
                    names.append([name,
                                  dept,
                                  year])  # Append name, dept, year only once
                i += 1
                cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 215, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
            cv2.imshow("Frame", frame)
            k = cv2.waitKey(1)
            if k == ord('q') or len(faces_data) == 100:
                break
        video.release()
        cv2.destroyAllWindows()

    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape(100, -1)

    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)

def main():
    collect_face_data()

if __name__ == "__main__":
    main()
