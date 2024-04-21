import tkinter as tk
from tkinter import messagebox, simpledialog
import os
import subprocess
import cv2
import pickle
import numpy as np
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

class InputDialog(tk.Toplevel):
    def __init__(self, parent, title):
        super().__init__(parent)
        self.title(title)
        self.geometry("300x200")
        self.name = ""
        self.dept = ""
        self.year = ""
        self.create_widgets()

    def create_widgets(self):
        name_label = tk.Label(self, text="Name:")
        name_label.pack(pady=5)
        self.name_entry = tk.Entry(self)
        self.name_entry.pack()

        dept_label = tk.Label(self, text="Department:")
        dept_label.pack(pady=5)
        self.dept_entry = tk.Entry(self)
        self.dept_entry.pack()

        year_label = tk.Label(self, text="Year:")
        year_label.pack(pady=5)
        self.year_entry = tk.Entry(self)
        self.year_entry.pack()

        submit_button = tk.Button(self, text="Submit", command=self.submit)
        submit_button.pack(pady=10)

    def submit(self):
        self.name = self.name_entry.get()
        self.dept = self.dept_entry.get()
        self.year = self.year_entry.get()
        self.destroy()

class HomePage(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Attendance Management System")
        self.geometry("1280x720")
        self.configure(bg="black")

        self.create_widgets()

    def create_widgets(self):
        self.canvas = tk.Canvas(self, width=1280, height=720, bg="black")
        self.canvas.pack()

        register_button = tk.Button(self, text="Register", font=("Arial", 24), command=self.register_faces)
        register_button.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

        take_attendance_button = tk.Button(self, text="Take Attendance", font=("Arial", 24), command=self.take_attendance)
        take_attendance_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

        show_attendance_button = tk.Button(self, text="Show Attendance", font=("Arial", 24), command=self.show_attendance)
        show_attendance_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

        attendance_count_button = tk.Button(self, text="Attendance Count", font=("Arial", 24), command=self.attendance_count)
        attendance_count_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

        self.update_background()

    def update_background(self):
        imgBackground = tk.PhotoImage(file="data/background.png")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgBackground)
        self.canvas.image = imgBackground

    def register_faces(self):
        input_dialog = InputDialog(self, "Enter Details")
        self.wait_window(input_dialog)
        name, dept, year = input_dialog.name, input_dialog.dept, input_dialog.year

        names = []  # Initialize an empty list to store names
        faces_data = []

        while len(names) < 100:
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
                        names.append([name, dept, year])  # Append name, dept, year only once
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

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

def take_attendance(self):
    # Initialize video capture from webcam
        

    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)[:100]

    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    knn = cv2.ml.KNearest_create()
    knn.train(FACES, cv2.ml.ROW_SAMPLE, LABELS)

    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%y")

    COL_NAMES = ['NAME', 'TIME']
    attendance_list = []

    exist = os.path.isfile(f"Attendance List/Attendance_{date}.csv")

    while True:
        ret, frame = video.read()  # Accessing video capture to read frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten()
            ret, output, _ = knn.findNearest(np.array([resized_img], dtype=np.float32), 5)
            output = int(output[0][0])
            ts = time.time()
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame, (x,y-40), (x+w, y+h), (50,50,255), -1)
            cv2.putText(frame, str(output), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
            attendance_list.append([str(output), str(timestamp)])
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    if exist:
        with open(f"Attendance List/Attendance_{date}.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            for row in attendance_list:
                writer.writerow(row)
    else:
        with open(f"Attendance List/Attendance_{date}.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(COL_NAMES)
            for row in attendance_list:
                writer.writerow(row)



def show_attendance(self):
        today_date = datetime.now().strftime("%d-%m-%y")
        file_path = f"Attendance List/Attendance_{today_date}.csv"
        if os.path.exists(file_path):
            subprocess.Popen(["notepad.exe", file_path])
        else:
            messagebox.showinfo("Attendance", "Attendance not taken yet for today.")

def attendance_count(self):
        today_date = datetime.now().strftime("%d-%m-%y")
        file_path = f"Attendance List/Attendance_{today_date}.csv"
        if os.path.exists(file_path):
            with open(file_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                count = sum(1 for row in reader) - 1  # subtracting header
            messagebox.showinfo("Attendance Count", f"Total Attendance Count for today: {count}")
        else:
            messagebox.showinfo("Attendance Count", "Attendance not taken yet for today.")

if __name__ == "__main__":
    app = HomePage()
    app.mainloop()
