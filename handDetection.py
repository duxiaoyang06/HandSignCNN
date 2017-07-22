import cv2
import numpy as np
import time
import imutils

class HandDetection:
    def __init__(self):
        self.init_handdetection()

    def init_handdetection(self):
        print("init")

    # Function to find angle between two vectors
    def Angle(self,v1,v2):
        dot = np.dot(v1,v2)
        x_modulus = np.sqrt((v1*v1).sum())
        y_modulus = np.sqrt((v2*v2).sum())
        cos_angle = dot / x_modulus / y_modulus
        angle = np.degrees(np.arccos(cos_angle))
        return angle

    # Function to find distance between two points in a list of lists
    def FindDistance(self,A,B): 
        return np.sqrt(np.power((A[0][0]-B[0][0]),2) + np.power((A[0][1]-B[0][1]),2)) 
     

    def HandRegion(self,image):
        print(image.shape)
        h,s,v = 100,100,100
        frame = image
        #Blur the image
        blur = cv2.blur(frame,(3,3))
        
        #Convert to HSV color space
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        
        #Create a binary image with where white will be skin colors and rest is black
        mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
        
        #Kernel matrices for morphological transformation    
        kernel_square = np.ones((11,11),np.uint8)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        
        #Perform morphological transformations to filter out the background noise
        #Dilation increase skin color area
        #Erosion increase skin color area
        dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
        erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
        dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
        filtered = cv2.medianBlur(dilation2,5)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
        dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        median = cv2.medianBlur(dilation2,5)
        ret,thresh = cv2.threshold(median,127,255,0)
        
        #different opencv version use this
        #contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
        #Find contours of the filtered frame
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
        
        #Draw Contours
        #cv2.drawContours(frame, cnt, -1, (122,122,0), 3)
        #cv2.imshow('Dilation',median)
        
        #Find Max contour area (Assume that hand is in the frame)
        max_area=100
        ci=0 
        if(len(contours)>0):
            for i in range(len(contours)):
                cnt=contours[i]
                area = cv2.contourArea(cnt)
                if(area>max_area):
                    max_area=area
                    ci=i  
                     
            #Largest area contour             
            cnts = contours[ci]

            #Find convex hull
            hull = cv2.convexHull(cnts)
            
            #Find convex defects
            hull2 = cv2.convexHull(cnts,returnPoints = False)
            defects = cv2.convexityDefects(cnts,hull2)
            
            #Get defect points and draw them in the original image
            FarDefect = []
            '''for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnts[s][0])
                end = tuple(cnts[e][0])
                far = tuple(cnts[f][0])
                FarDefect.append(far)
                cv2.line(frame,start,end,[0,255,0],1)
                cv2.circle(frame,far,10,[100,255,255],3)
            '''
            #Find moments of the largest contour
            moments = cv2.moments(cnts)
            
            #Central mass of first order moments
            if moments['m00']!=0:
                cx = int(moments['m10']/moments['m00']) # cx = M10/M00
                cy = int(moments['m01']/moments['m00']) # cy = M01/M00
            centerMass=(cx,cy)    
            
            #Draw center mass
            #cv2.circle(frame,centerMass,7,[100,0,255],2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(frame,'Center',tuple(centerMass),font,2,(255,255,255),2)     
            
            #Distance from each finger defect(finger webbing) to the center mass
            distanceBetweenDefectsToCenter = []
            for i in range(0,len(FarDefect)):
                x =  np.array(FarDefect[i])
                centerMass = np.array(centerMass)
                distance = np.sqrt(np.power(x[0]-centerMass[0],2)+np.power(x[1]-centerMass[1],2))
                distanceBetweenDefectsToCenter.append(distance)
            
            #Get an average of three shortest distances from finger webbing to center mass
            sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
            AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])
         
            #Get fingertip points from contour hull
            #If points are in proximity of 80 pixels, consider as a single point in the group
            finger = []
            for i in range(0,len(hull)-1):
                if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
                    if hull[i][0][1] <500 :
                        finger.append(hull[i][0])
            
            #The fingertip points are 5 hull points with largest y coordinates  
            finger =  sorted(finger,key=lambda x: x[1])   
            fingers = finger[0:5]
            
            #Calculate distance of each finger tip to the center mass
            fingerDistance = []
            for i in range(0,len(fingers)):
                distance = np.sqrt(np.power(fingers[i][0]-centerMass[0],2)+np.power(fingers[i][1]-centerMass[0],2))
                fingerDistance.append(distance)
            
            #Finger is pointed/raised if the distance of between fingertip to the center mass is larger
            #than the distance of average finger webbing to center mass by 130 pixels
            result = 0
            for i in range(0,len(fingers)):
                if fingerDistance[i] > AverageDefectDistance+130:
                    result = result +1
            
            #Print bounding rectangle
            x,y,w,h = cv2.boundingRect(cnts)
            #green square rectangel
            #img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            '''
            #COLOR SEGMENTATION
            #skin limits
            lower = np.array([0, 10, 10], dtype = "uint8")
            upper = np.array([80, 255, 255], dtype = "uint8")
            #frame = imutils.resize(frame, width = 400)
            frame = frame[y:y+h, x:x+w]
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            skinMask = cv2.inRange(converted, lower, upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            skinMask = cv2.erode(skinMask, kernel, iterations = 2)
            skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
            skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
            skin = cv2.bitwise_and(frame, frame, mask = skinMask)
            cv2.imshow('Dilation', np.hstack([frame, skin]))
            '''

            sub_image = frame[y:y+h, x:x+w]
            #gray = cv2.cvtColor(sub_image,cv2.COLOR_BGR2GRAY)
            #ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            #cv2.drawContours(frame,[hull],-1,(255,255,255),2)
            print("subimage")
            resized_image = cv2.resize(sub_image, (52, 52)) 
            return x,y,w,h,resized_image
        print("NONE")
        return None