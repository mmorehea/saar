import cv2
import numpy as np
import os
import sys
import glob
import code
import tifffile
from timeit import default_timer as timer
import ConfigParser
from scipy import ndimage

def adjustThresh(originalImg, value):
	ret,thresh1 = cv2.threshold(originalImg, int(value), 255, cv2.THRESH_BINARY)
	kernel = np.ones((3,3),np.uint8)

	thresh1 = cv2.dilate(thresh1, kernel, 1)

	thresh1 = np.uint8(ndimage.morphology.binary_fill_holes(thresh1))
	ret,thresh1 = cv2.threshold(thresh1, 0, 255, cv2.THRESH_BINARY)

	thresh1 = cv2.erode(thresh1, kernel, 1)

	#thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
	#thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
	return thresh1

def adjustContours(kernelImg, kernelSize):
	# kernelImg = cv2.cvtColor(kernelImg, cv2.COLOR_BGR2GRAY)
	# print kernelImg.dtype
	blank = np.zeros(kernelImg.shape)

	if cv2.__version__[0] == '3':
		contourImage, contours, hierarchy = cv2.findContours(kernelImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
	else:
		contours, hierarchy = cv2.findContours(kernelImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
	cv2.drawContours(blank, contours, -1, (255,255,255), -1)
	kernel = np.ones((kernelSize,kernelSize),np.uint8)

	#blank = cv2.dilate(blank, kernel, 1)
	#blank = cv2.erode(blank, kernel, 2)
	return np.uint8(blank)

def nothing(x):
    pass

def threshVis(img):
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

	return oldThresh, threshImg

def noiseVis(threshImg):
	oldKernel = 2
	ks = oldKernel
	cv2.namedWindow('image')
	threshImg = np.uint8(threshImg)

	kernelImg = np.uint8(threshImg)
	while(1):
		try:
			kernelImg = cv2.resize(kernelImg, (1900, 1200))
			cv2.imshow('image', kernelImg)
		except:
			print 'WARNING: cv2 did not read the image correctly'
		k = cv2.waitKey(1)
		if k == 32:
			break
		if k == 112:
			ks += 1
		if k == 111:
			ks -= 1
		if k == 113:
			cv2.destroyAllWindows()
			sys.exit(0)
		# get current positions of four trackbars

		if (ks != oldKernel):
			print ks
			oldKernel = ks
			kernelImg = cv2.morphologyEx(threshImg, cv2.MORPH_OPEN, np.ones((ks,ks)))
			ret,kernelImg = cv2.threshold(kernelImg, 0, 255, cv2.THRESH_BINARY)

	cv2.destroyAllWindows()
	return oldKernel, kernelImg

def contourVis(img, threshImg):
	oldKernel = 2
	ks = oldKernel
	cv2.namedWindow('image')

	kernelImg = np.uint8(threshImg)
	while(1):
		try:
			kernelImg = cv2.resize(kernelImg, (1900, 1200))
			cv2.imshow('image', kernelImg)
		except:
			print 'WARNING: cv2 did not read the image correctly'
		k = cv2.waitKey(1)
		if k == 32:
			break
		if k == 112:
			ks += 1
		if k == 111:
			ks -= 1
		if k == 113:
			cv2.destroyAllWindows()
			sys.exit(0)
		# get current positions of four trackbars

		if (ks != oldKernel):
			print ks
			oldKernel = ks
			kernelImg = adjustContours(threshImg, ks)

	cv2.destroyAllWindows()
	return oldKernel, kernelImg

def saveAndQuit(oldThresh, noiseKernel):
	print "Writing configuration file..."
	cfgfile = open("saar.ini",'w')
	Config = ConfigParser.ConfigParser()
	Config.add_section('Options')
	Config.set('Options','Threshold Value', oldThresh)
	Config.set('Options','Remove Noise Kernel Size', noiseKernel)
	Config.write(cfgfile)
	cfgfile.close()

def main():
	emFolderPath = sys.argv[1]
	emPaths = sorted(glob.glob(emFolderPath +'*'))

	em = emPaths[0]
	img = cv2.imread(em, -1)


	while True:
		print("SAAR MENU")
		print("1. Threshold")
		print("2. Remove Noise")
		print("3. Save and Quit")
		choice = raw_input(">")
		if choice=='1':
			oldThresh, threshImg = threshVis(img)
		elif choice=='2':
			noiseKernel, threshImg = noiseVis(threshImg)
		elif choice=='3':
			saveAndQuit(oldThresh, noiseKernel)
			break
		else:
			continue




if __name__ == "__main__":
	main()
