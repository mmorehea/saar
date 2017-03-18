import cv2
import numpy as np
import os
import sys
import glob
import code
import tifffile
from timeit import default_timer as timer
import ConfigParser

def adjustThresh(originalImg, value):
	ret,thresh1 = cv2.threshold(originalImg, int(value), 255, cv2.THRESH_BINARY)
	kernel = np.ones((2,2),np.uint8)
	#thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
	#thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
	return thresh1

def ajdustContours(kernelImg, kernelSize):
	blank = np.zeros(kernelImg.shape)

	if cv2.__version__[0] == '3':
		contourImage, contours, hierarchy = cv2.findContours(kernelImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
	else:
		contours, hierarchy = cv2.findContours(kernelImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
	cv2.drawContours(blank, contours, -1, (255,255,255), -1)
	kernel = np.ones((kernelSize,kernelSize),np.uint8)
	blank = cv2.morphologyEx(blank, cv2.MORPH_CLOSE, kernel)
	blank = cv2.erode(blank, kernel, 1)
	return blank

def nothing(x):
    pass

def main():
	emFolderPath = sys.argv[1]
	emPaths = sorted(glob.glob(emFolderPath +'*'))

	em = emPaths[0]
	img = cv2.imread(em, -1)
	oldThresh = 200
	cv2.namedWindow('image')

	# create trackbars for picking threshold
	cv2.createTrackbar('Threshold', 'image', 0, 255, nothing)
	threshImg = img
	while(1):
		try:
			threshImg = cv2.resize(threshImg, (1900, 1200))
			cv2.imshow('image', threshImg)
		except:
			print 'WARNING: cv2 did not read the image correctly'
		k = cv2.waitKey(1)
		if k == 32:
			break
		# get current positions of four trackbars
		r = cv2.getTrackbarPos('Threshold','image')
		if (r != oldThresh):
			oldThresh = r
			threshImg = adjustThresh(img, r)

	cv2.destroyAllWindows()
	oldKernel = 0
	cv2.namedWindow('image')

	# create trackbars for picking kernel size
	cv2.createTrackbar('Kernel Size', 'image', 2, 50, nothing)
	kernelImg = img
	while(1):
		try:
			kernelImg = cv2.resize(kernelImg, (1900, 1200))
			cv2.imshow('image', kernelImg)
		except:
			print 'WARNING: cv2 did not read the image correctly'
		k = cv2.waitKey(1)
		if k == 32:
			break
		# get current positions of four trackbars
		ks = cv2.getTrackbarPos('Kernel Size','image')
		if (ks != oldKernel):
			oldKernel = ks
			kernelImg = adjustContours(img, ks)

	cv2.destroyAllWindows()

	print "Writing configuration file..."
	cfgfile = open("saar.ini",'w')
	Config = ConfigParser.ConfigParser()
	Config.add_section('Options')
	Config.set('Options','Threshold Value', oldThresh)
	Config.set('Options','Kernel Size', oldKernel)
	Config.write(cfgfile)
	cfgfile.close()


if __name__ == "__main__":
	main()
