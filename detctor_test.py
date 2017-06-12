# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 00:36:54 2017

@author: Black丶Light
"""

import cv2
import pickle
import sqlite3
import numpy as np
from PIL import Image

recongnizer = cv2.createLBPHFaceRecognizer()
recongnizer.load("recognizer\\trainningData.yml")
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
path = "dataSet"


def getProfile(id):
    conn = sqlite3.connect("FaceBase.db")
    cmd  = "SELECT * FROM People WHERE ID=" + str(id)
    cursor=conn.execute(cmd)
    profile = 0
    for row in cursor:
        profile = row
    conn.close()
    return profile

cam = cv2.VideoCapture(0)
cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,280)
cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,240)
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX,1,1,0,1,1)
#font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,1, .5,0,2,1)
profiles = []
while True:
    ret,im = cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray , scaleFactor = 1.2 ,minNeighbors = 5, minSize = (100,100), flags = cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
            nbr_predicted,conf=recongnizer.predict(gray[y:y+h,x:x+w])
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
            profile = getProfile(nbr_predicted)
            if(profile != None):
                cv2.cv.PutText(cv2.cv.fromarray(im), "Name:" + str(profile[1]), (x, y + h + 20), font, (0, 0, 255))
                cv2.cv.PutText(cv2.cv.fromarray(im), "Age: " + str(profile[2]), (x, y + h + 50), font, (0, 0, 255))
            id,conf = recongnizer.predict(gray[y:y+h,x:x+w])
            print id
            cv2.cv.PutText(cv2.cv.fromarray(im),str(id),(x,y + h),font,255)
    cv2.imshow("Face",im)
    cv2.waitKey(10)
    if(cv2.waitKey(1) == ord('q')):
            break
cam.release()
cv2.destroyAllWindows()