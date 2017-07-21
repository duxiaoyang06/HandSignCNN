import cv2
import numpy as np
import glob
import uuid
#import skimage.io
#from util import histogram_equalization
from scipy.ndimage import zoom
#from skimage.transform import resize
import random
import cv2
import numpy as np
#from matplotlib import pyplot as plt
#import dlib
#from project_face import frontalizer
#import matplotlib.pyplot as plt

IMAGE_WIDTH = 52
IMAGE_HEIGHT = 52

class hand_detector():
    def __init__(self):
        self.hand_cascade = cv2.CascadeClassifier('aGest.xml')

    def hand_detect_single(self,image,isPath):
        if isPath == True:
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED) 
        else:
            img = image
        
        
        #cv2.waitKey(666)
        #img = histogram_equalization(img)
        #gray_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        hands = self.hand_cascade.detectMultiScale(img)
        print(hands)
        for (x,y,w,h) in hands:
            print(x)
            print(y)
            print(w)
            print(h)
            roi_gray = img[y:y+h, x:x+w]
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),1,0)
            #cv2.imshow("Display window", img );
            #cv2.waitKey(6666)
            rectan = dlib.rectangle(long(x),long(y),long(x+w),long(y+h))
            crop_img_resized = cv2.resize(roi_gray, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
            #print(x)
            return crop_img_resized,rectan.left(),rectan.top(),rectan.right(),rectan.bottom()
        else:
            return None,-1,-1,-1,-1

    def hand_detect_bulk(self,input_folder,output_folder):
        transformed_data_set = [img for img in glob.glob(input_folder+"/*jpg")]

        for in_idx, img_path in enumerate(transformed_data_set):
            mouth,x,y,w,h = self.mouth_detect_single(img_path,True)
            if 'showingteeth' in img_path:
                guid = uuid.uuid4()
                uid_str = guid.urn
                str_guid = uid_str[9:]
                path = output_folder+"/"+str_guid+"_showingteeth.jpg"
                cv2.imwrite(path,mouth)
            else:
                guid = uuid.uuid4()
                uid_str = guid.urn
                str_guid = uid_str[9:]
                path = output_folder+"/"+str_guid+".jpg"
                cv2.imwrite(path,mouth)

    def negative_image(self,imagem):
        imagem = (255-imagem)
        return imagem

    def adaptative_threashold(self,input_img_path):
        img = cv2.imread(input_img_path,0)
        img = cv2.medianBlur(img,3)
        ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)
        #cv2.imwrite("../img/output_test_img/hmouthdetectsingle_adaptative.jpg",th3)
        return th3