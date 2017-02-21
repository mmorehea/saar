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

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import functools


def adjustThresh(originalImg, value):
	ret,thresh1 = cv2.threshold(originalImg, int(value), 255, cv2.THRESH_BINARY)
	kernel = np.ones((2,2),np.uint8)
	#thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
	thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
	return thresh1

def nothing(x):
    pass

def processEntireStack(path, threshValue):
	pool = ThreadPool(8) 
	emFolderPath = "cropedEM/"
	emPaths = sorted(glob.glob(emFolderPath +'*'))
	emImages = [cv2.imread(emPaths[z], -1) for z in xrange(len(emPaths))]
	result = pool.map(functools.partial(adjustThresh, value = threshValue), emImages)
	result2 = pool.map(contourAndErode, result)
	print "length of result: " + str(len(result2))
	return result2

def contourAndErode(threshImg):
	blank = np.zeros(threshImg.shape)
	kernel = np.ones((3,3),np.uint8)
	if cv2.__version__[0] == '3':
		contourImage, contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
	else:
		contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
	cv2.drawContours(blank, contours, -1, (255,255,255), -1)
	kernel = np.ones((2,2),np.uint8)	
	blank = cv2.morphologyEx(blank, cv2.MORPH_CLOSE, kernel)

	blank = cv2.erode(blank, kernel, 1)
	return blank

def cleanLabels(img):
	kernel = np.ones((2,2),np.uint8)

	uniqueLabels = np.unique(img)
	for lab in uniqueLabels:
		blankImg = np.zeros(img.shape)
		indices = np.where(img==lab)
		blankImg[indices] = 99999
		img[indices] = 0

		blankImg = cv2.dilate(blankImg, kernel, 2)
		blankImg = cv2.erode(blankImg, kernel, 1)
		img[np.nonzero(blankImg)] = lab

def main():
	em = "cropedEM/Crop_mendedEM-0000.tiff"
	img = cv2.imread(em, -1)
	oldThresh = 200
	cv2.namedWindow('image')

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
	blank = np.dstack(blank)
	endContourStack = timer() - startMain
	start = timer()
	print "Finding connection..."
	labels = nd.measurements.label(blank)
	labels = labels[0]
	labels = np.uint16(labels) 
	endConnections = timer() - start

	print "Cleaning labels..."
	pool = ThreadPool(8)
	labels = np.dstack(pool.map(cleanLabels, np.dsplit(labels, labels.shape[2])))
	endClean = timer() - start

	print "Writing file..."
	start = timer()
	for each in xrange(labels.shape[2]):
		print each
		img = labels[:,:,each]
		tifffile.imsave("out2/" + str(each) + '.tif', img)
	endWritingFile = timer() - start

	endTime = timer()

	with open('runStats_multi2.txt', 'w') as f:
		f.write('Run Stats \n')
		f.write('Run start: ' + str(startMain) + '\n')
		f.write('Total time: '+ str(endTime - startMain) + '\n')
		f.write('Contour time: ' + str(endContourStack) + '\n')
		f.write('Connection time: ' + str(endConnections) + '\n')
		f.write('Clean time: ' + str(endClean) + '\n')
		f.write('Writing file time: ' + str(endWritingFile) + '\n')
		f.write('Total number of labels: ' + str(len(np.unique(labels))))


cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
