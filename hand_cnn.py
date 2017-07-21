import numpy as np
import sys
import glob
import uuid
import cv2
#from util import transform_img
from handDetect import hand_detector
import os
import shutil
#from util import histogram_equalization

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

class hand_cnn:

	def __init__(self):
		self.init_net()

	def init_net(self):
		print("init")

	def mean_blob_fn(self):
		print("mean")

	def predict(self,image,hand_detector):
		img = image
		mouth_pre,x,y,w,h = hand_detector.hand_detect_single(image,False)
		return x,y,w,h