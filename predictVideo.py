import numpy as np
import sys
import glob
import uuid
import cv2
from handDetection import HandDetection
import os
import shutil
import math
from hand_cnn import HandCnn

hd = HandDetection()

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
vc.set(3,600)
vc.set(4,600)
#vc.set(5,100)
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

hand_cnn_instance = HandCnn()

while rval:
	rval, frame = vc.read()
	copy_frame = frame.copy()
	_x,_y,_w,_h,handRegionImg = hd.HandRegion(copy_frame)
	#MANUALLY FILTER HAND REGIONS...
	imgray = cv2.cvtColor(handRegionImg,cv2.COLOR_BGR2GRAY)
	#Code for histogram equalization
	equ = cv2.equalizeHist(imgray)
	predictions = hand_cnn_instance.predict_single(equ)
	print(predictions)
	#p1, p2
	cv2.rectangle(frame, (_x,_y),(_w,_h),(0,255,0),4,0)
	cv2.imshow("preview", frame)
	cv2.waitKey(1)
cv2.destroyWindow("preview")