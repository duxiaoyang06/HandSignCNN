from handDetection import HandDetection
import cv2
from util import shear_image
from util import rotate_image
from util import image_rotated_cropped
import glob
import uuid
import os
import shutil
from shutil import copyfile





#img = cv2.imread("test.jpg")
#handRegionImg = hd.HandRegion(img)
#print(handRegionImg)

#cv2.imshow('Dil',handRegionImg)
#cv2.waitKey(10000)
#print("finish")

#1. read original dataset and extract hand regions, output the result to original_data_augmented
#2. read original_data_augented and replicate on this same folder by shearing and rotating
#3. apply post processing(optional)
#4. copy result to training_data



#Augment data by x8 times by making scalings and small rotations




#1. read original dataset and extract hand regions, output the result to original_data_augmented
IMAGE_WIDTH = 52
IMAGE_HEIGHT = 52

input_folder = "original_data"
output_folder_handregion ="original_data_handregion"
output_folder_augmented ="original_data_augmented"
output_folder_training ="training_data"
classes = ['A', 'B','C']


#purge data
shutil.rmtree(output_folder_handregion)
shutil.rmtree(output_folder_augmented)
shutil.rmtree(output_folder_training)
#create class folders in each dir
for cls in classes:
	os.makedirs(output_folder_handregion+"/"+cls)
	os.makedirs(output_folder_augmented+"/"+cls)
	os.makedirs(output_folder_training+"/"+cls)



hd = HandDetection()
generate_random_filename = 1
print("TRAINING GENERATOR...")

for cls in classes:
	input_data_set = [img for img in glob.glob(input_folder+"/"+cls+"/"+"*jpg")]
	print("INPUTDATASET")
	print(input_data_set)
	for in_idx, img_path in enumerate(input_data_set):
		#extract hand regions
		file_name = os.path.splitext(os.path.basename(img_path))[0]
		print(file_name)
		path = output_folder_handregion+"/"+cls+"/"+file_name+".jpg"
		img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
		_x,_y,_w,_h,handRegionImg = hd.HandRegion(img)
		#MANUALLY FILTER HAND REGIONS...
		imgray = cv2.cvtColor(handRegionImg,cv2.COLOR_BGR2GRAY)
		#Code for histogram equalization
		equ = cv2.equalizeHist(imgray)
		print("OUTPUT")
		print(path)
		cv2.imwrite(path,equ)


#DATA AUGMENTATION
for cls in classes:
	input_data_set = [img for img in glob.glob(output_folder_handregion+"/"+cls+"/"+"*jpg")]
	for in_idx, img_path in enumerate(input_data_set):
		file_name = os.path.splitext(os.path.basename(img_path))[0]
		print(file_name)
		augmentation_number = 8
		initial_rot = -20
		#save original too
		path = output_folder_augmented+"/"+cls+"/"+file_name+".jpg"
		copyfile(img_path, path)
		for x in range(1, augmentation_number):
			rotation_coeficient = x
			rotation_step=5
			total_rotation=initial_rot+rotation_step*rotation_coeficient
			#print(total_rotation)
			mouth_rotated = image_rotated_cropped(img_path,total_rotation)
			#resize to 50 by 50
			mouth_rotated = cv2.resize(mouth_rotated, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
			if generate_random_filename == 1:
				guid = uuid.uuid4()
				uid_str = guid.urn
				str_guid = uid_str[9:]
				path = ""
				if 'showingteeth' in img_path:
				    path = output_folder_augmented+"/"+str_guid+"_showingteeth.jpg"
				else:
				    path = output_folder_augmented+"/"+cls+"/"+str_guid+".jpg"
				cv2.imwrite(path,mouth_rotated)
			else:
				path = ""
				if 'showingteeth' in img_path:
				    path = output_folder_augmented+"/"+file_name+"_rotated"+str(x)+"_showingteeth.jpg"
				else:
				    path = output_folder_augmented+"/"+cls+"/"+file_name+"_rotated"+str(x)+".jpg"
				cv2.imwrite(path,mouth_rotated)

#COPY TO TRAINING FOLDER
for cls in classes:
	input_data_set = [img for img in glob.glob(output_folder_augmented+"/"+cls+"/"+"*jpg")]
	for in_idx, img_path in enumerate(input_data_set):
		file_name = os.path.splitext(os.path.basename(img_path))[0]
		print(file_name)
		augmentation_number = 8
		initial_rot = -20
		#save original too
		path = output_folder_training+"/"+cls+"/"+file_name+".jpg"
		copyfile(img_path, path)