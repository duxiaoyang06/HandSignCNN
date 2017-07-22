import numpy as np
import sys
import glob
import uuid
import cv2
#from util import transform_img
from handDetect import hand_detector
from handDetection import HandDetection
import os
import shutil
import tensorflow as tf
import os
#from util import histogram_equalization

IMAGE_WIDTH = 52
IMAGE_HEIGHT = 52
num_channels = 1

class HandCnn:

	def __init__(self):
		self.sess=tf.Session()    
		self.saver = tf.train.import_meta_graph('my_test_model_iteration_320.meta')
		self.saver.restore(self.sess,'my_test_model_iteration_320')
		self.graph = tf.get_default_graph()
		self.x = self.graph.get_tensor_by_name("x:0")
		self.y_conv = self.graph.get_tensor_by_name("y_conv:0")
		self.keep_prob = self.graph.get_tensor_by_name("keep_prob:0")
		self.y_conv_cls = self.graph.get_tensor_by_name("y_conv_cls:0")
		self.img_size = 52
		self.hd = HandDetection()

	@property
  	def x(self):
  		return self.x

	def predict_batch(self,dir_path):
		images = []
		path = os.path.join(dir_path, '*g')
		files = glob.glob(path)
		for fl in files:
			#print(fl)
			image = cv2.imread(fl,cv2.IMREAD_UNCHANGED)
			#handmade feature...
			#print(image)
			_x,_y,_w,_h,image = self.hd.HandRegion(image)
			image = cv2.resize(image, (self.img_size, self.img_size), cv2.INTER_LINEAR)
			images.append(image)
		images = np.array(images)
		train_batch_size = len(files)
		img_size_flat = self.img_size * self.img_size * num_channels
		x_batch = images;
		x_batch = x_batch.reshape(train_batch_size, img_size_flat)
		feed_dict = {self.x:x_batch,self.keep_prob:1}
		#print sess.run(y_conv,feed_dict)
		prediction = self.sess.run(self.y_conv_cls,feed_dict)
		return prediction

	def predict_single(self,image):
		images = []
		print(image.shape)
		#image = cv2.resize(image, (self.img_size, self.img_size), cv2.INTER_LINEAR)
		images.append(image)
		images = np.array(images)
		train_batch_size = 1
		img_size_flat = self.img_size * self.img_size * num_channels
		print(self.img_size)
		print(img_size_flat)
		x_batch = images;
		x_batch = x_batch.reshape(train_batch_size, img_size_flat)
		feed_dict = {self.x:x_batch,self.keep_prob:1}
		#print sess.run(y_conv,feed_dict)
		prediction = self.sess.run(self.y_conv_cls,feed_dict)
		return prediction
