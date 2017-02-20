import cv2
import numpy as np
import os
import sys
import glob
import code
import tifffile
from timeit import default_timer as timer

from scipy import ndimage as nd

from skimage import measure
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

import matplotlib.pyplot as plt


def adjustThresh(originalImg, value):
	ret,thresh1 = cv2.threshold(originalImg, int(value), 255, cv2.THRESH_BINARY)
	return thresh1

def nothing(x):
    pass

def processEntireStack(path, threshValue):
	emFolderPath = "cropedEM/"
	emPaths = sorted(glob.glob(emFolderPath +'*'))
	emImages = [cv2.imread(emPaths[z], -1) for z in xrange(len(emPaths))]
	processedStack = []
	for ii, each in enumerate(emImages):
		print str(ii) + " / " + str(len(emImages))
		threshImg = adjustThresh(each, threshValue)
		processedImg = contourAndErode(each, threshImg)
		processedStack.append(processedImg)

	processedStack = np.dstack(processedStack)
	return processedStack


def contourAndErode(img, threshImg):
	blank = np.zeros(img.shape)
	kernel = np.ones((3,3),np.uint8)
	if cv2.__version__[0] == '3':
		contourImage, contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
	else:
		contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
	cv2.drawContours(blank, contours, -1, (255,255,255), -1)
	#blank = cv2.morphologyEx(blank, cv2.MORPH_CLOSE, kernel)
	kernel = np.ones((2,2),np.uint8)
	blank = cv2.erode(blank, kernel, 1)
	return blank

def main():
	

	#emFolderPath = "cropedEM/"
	#emPaths = sorted(glob.glob(emFolderPath +'*'))

	#emImages = [cv2.imread(emPaths[z], -1) for z in xrange(len(emPaths))]
	#emImages = np.dstack(emImages)
	em = "cropedEM/Crop_mendedEM-0000.tiff"
	img = cv2.imread(em, -1)
	oldThresh = 200
	cv2.namedWindow('image')
	#code.interact(local=locals())

	# create trackbars for picking threshold
	cv2.createTrackbar('Threshold', 'image', 0, 255, nothing)
	threshImg = img
	while(1):
		cv2.imshow('image', threshImg)
		k = cv2.waitKey(1)
		if k == 32:
			break


			# get current positions of four trackbars
		r = cv2.getTrackbarPos('Threshold','image')
		if (r != oldThresh):
			oldThresh = r
			threshImg = adjustThresh(img, r)



	startMain = timer()
	print "Contouring entire stack..."
	blank = processEntireStack('ok', oldThresh)
	endContourStack = timer() - startMain
	start = timer()
	print "Finding connection..."
	labels = nd.measurements.label(blank)
	labels = labels[0]
	endConnections = timer() - start
	
	print "Writing file..."
	start = timer()
	for each in range(labels.shape[2]):
		print each
		img = labels[:,:,each]
		tifffile.imsave("out/" + str(each) + '.tif', img)
	endWritingFile = timer() - start

	endTime = timer()

	with open('runStats.txt', 'w') as f:
		f.write('Run Stats \n')
		f.write('Run start: ' + str(startMain) + '\n')
		f.write('Total time: '+ str(endTime - startMain) + '\n')
		f.write('Contour time: ' + str(endContourStack) + '\n')
		f.write('Connection time: ' + str(endConnections) + '\n')
		f.write('Writing file time: ' + str(endWritingFile) + '\n')
		f.write('Total number of labels: ' + str(np.unique(labels)))











cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
