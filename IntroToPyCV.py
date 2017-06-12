import cv2
import numpy as np 
import sqlite3

faceDetect   = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,280)
cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,240)
id    = raw_input('enter user id:    ')
name  = raw_input("enter user name:  ")
local = raw_input("enter user local: ")
cmd = sqlite3.connect("FaceBase.db")
v = (id, name,local)
ins = "insert into People values(?,?,?);"
cmd.execute(ins, v)
sampleNum = 0
cmd.commit()
cmd.close()
while(True):
	ret,img=cam.read()
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces=faceDetect.detectMultiScale(gray,1.3,5)
	for(x,y,w,h) in faces:
		sampleNum=sampleNum+1
		cv2.imwrite("dataSet/User." + str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
	cv2.imshow("Face",img)
	cv2.waitKey(1)
	if(sampleNum > 20):
			break 

        
cam.release()
cv2.destroyAllWindows()