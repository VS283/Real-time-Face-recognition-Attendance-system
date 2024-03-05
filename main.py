import cv2
import os
import pickle
import face_recognition
import numpy as np
from datetime import datetime, timedelta
import csv


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('Resources/background.png')

# Importing the mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
    # print(len(imgModeList))

# Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
# print(studentIds)
print("Encode File Loaded")

Attendence = {}
modetype = 0

with open('attendance.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Student ID', 'Time'])

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modetype]

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print("matches", matches)
        # print("faceDis", faceDis)

        matchIndex = np.argmin(faceDis)
        # print("Match Index", matchIndex)

        if matches[matchIndex]:
            # print("Known Face Detected")
            # print(studentIds[matchIndex])
            id = studentIds[matchIndex]
            if id in Attendence and (datetime.now() - Attendence[id]) < timedelta(seconds=10):
                modetype = 2
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modetype]
                # print('am')
                if modetype == 2 :
                    timedelta(modetype, seconds=5)
                    modetype = 0
            else:
                # Mark attendance and set the state to "attendance_mark"
                Attendence[id] = datetime.now()
                # Date_Time = Attendence[id].strftime("%Y-%m-%d %H:%M:%S")
                # print(LastAttendence)
                # Append the data to CSV file
                modetype = 1
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modetype]
                # print('p')
                with open('attendance.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([id, datetime.now(), 'Present'])

        else:
            modetype = 0
            imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modetype]
            print('0')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # cv2.imshow("webcam", img)
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)
