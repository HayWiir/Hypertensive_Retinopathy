
# coding: utf-8

# import the necessary packages
from __future__ import print_function
import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import argparse
import imutils
from imutils import contours
import cv2
from skimage import measure
from matplotlib import pyplot as plt
import os

directory = "/home/kunal/Documents/Hypertensive_Retinopathy/Dataset"
for filename in os.listdir(directory):

    if filename.endswith(".jpg"): 
    	image_path = os.path.join(directory, filename)

    	image = cv2.imread(image_path)

    	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    	blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    	thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

    	# perform a series of erosions and dilations to remove
    	# any small blobs of noise from the thresholded image
    	thresh = cv2.erode(thresh, None, iterations=2)
    	thresh = cv2.dilate(thresh, None, iterations=4)

    	labels = measure.label(thresh, neighbors=8, background=0)
    	mask = np.zeros(thresh.shape, dtype="uint8")

    	for label in np.unique(labels):
    		if label == 0:
    			continue

    		labelMask = np.zeros(thresh.shape, dtype="uint8")
    		labelMask[labels == label] = 255
    		numPixels = cv2.countNonZero(labelMask)

    		if numPixels > 300:
    			mask = cv2.add(mask, labelMask)

    	try:
    		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    		cnts = contours.sort_contours(cnts)[0]
    	except ValueError:
    		print(filename)
    		continue

    	for (i, c) in enumerate(cnts):
    		(x, y, w, h) = cv2.boundingRect(c)
    		((cX, cY), radius) = cv2.minEnclosingCircle(c)
    		cv2.circle(image, (int(cX), int(cY)), int(radius),(0, 0, 255), 3)
    		cv2.putText(image, "#{}".format(i + 1), (x, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    	newfilename = "LBS_"+filename
    	cv2.imwrite('../Labels/'+newfilename, image)	

	    # try:	
	    # 	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	    # 	
	    # 	

	    # 	
	    # 		(x, y, w, h) = cv2.boundingRect(c)
	    # 		
	    # 		
	    # 		
	    # 	

	    # except ValueError:
	    # 	print(filename)	



    		


