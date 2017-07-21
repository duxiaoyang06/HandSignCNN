import numpy as np
import sys
import glob
import uuid
import cv2
from handDetect import hand_detector
import os
import shutil
import math
from hand_cnn import hand_cnn

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
vc.set(3,200)
vc.set(4,200)
#vc.set(5,100)
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

hand_detector_instance = hand_detector()
hand_cnn_instance = hand_cnn()

size = cv2.getTextSize("Showing teeth", cv2.FONT_HERSHEY_PLAIN, 2, 1)[0]
x,y = (50,250)

xf,yf,wf,hf = hand_cnn_instance.predict(None,hand_detector_instance)


'''
while rval:
	rval, frame = vc.read()
	copy_frame = frame.copy()
	xf,yf,wf,hf = hand_cnn_instance.predict(copy_frame,hand_detector_instance)
	if(xf != -1):
		print(xf)
		print(yf)
		print(wf)
		print(hf)
		cv2.rectangle(frame, (xf,yf),(wf,hf),(0,255,0),4,0)
		cv2.rectangle(frame, (xf-2,yf-25),(wf+2,yf),(0,255,0),-1,0)
		cv2.rectangle(frame, (xf-2,hf),(xf+((wf-xf)/2),hf+25),(0,255,0),-1,0)
		cv2.putText(frame, "Teeth!!",(xf,hf+14),cv2.FONT_HERSHEY_PLAIN,1.2,0,2)
		cv2.putText(frame, str(prob_round)+"%",(xf,yf-10),cv2.FONT_HERSHEY_PLAIN,1.2,0,2)
		#out.write(frame)
		print ("SHOWING HAND!!!")
	
	cv2.imshow("preview", frame)
	cv2.waitKey(1)
cv2.destroyWindow("preview")'''