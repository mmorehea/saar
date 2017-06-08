import cv2
import numpy as np
import os
import sys
import glob
import code
import tifffile
from timeit import default_timer as timer
import ConfigParser
from scipy import ndimage as nd

def adjustThresh(originalImg, value):
	ret,thresh1 = cv2.threshold(originalImg, int(value), 255, cv2.THRESH_BINARY)
	kernel = np.ones((3,3),np.uint8)
	thresh1 = cv2.dilate(thresh1, kernel, 1)
	thresh1 = np.uint8(nd.morphology.binary_fill_holes(thresh1))
	ret,thresh1 = cv2.threshold(thresh1, 0, 255, cv2.THRESH_BINARY)
	thresh1 = cv2.erode(thresh1, kernel, 1)
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
		k = cv2.waitKey(1)
		if k == 32:
			break
		try:
			# threshImg = cv2.resize(threshImg, (1900, 1200))
			cv2.imshow('image', threshImg)
		except:
			print 'WARNING: cv2 did not read the image correctly'

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
	cv2.createTrackbar('Kernel Size for Noise Removal', 'image', 1, 10, nothing)
	while(1):
		k = cv2.waitKey(1)
		if k == 32:
			break
		try:
			# kernelImg = cv2.resize(kernelImg, (1900, 1200))
			cv2.imshow('image', kernelImg)
		except:
			print 'WARNING: cv2 did not read the image correctly'
		ks = cv2.getTrackbarPos('Kernel Size for Noise Removal', 'image')
		# get current positions of four trackbars

		if (ks != oldKernel):
			# print ks
			oldKernel = ks
			kernelImg = cv2.morphologyEx(threshImg, cv2.MORPH_OPEN, np.ones((ks,ks)))
			ret,kernelImg = cv2.threshold(kernelImg, 0, 255, cv2.THRESH_BINARY)
			# kernelImg = cv2.erode(kernelImg, (ks,ks), 6)

	cv2.destroyAllWindows()
	return oldKernel, kernelImg

def sizeVis(img):

	sizeRange = [0,99]
	cv2.namedWindow('image')

	cv2.createTrackbar('Lowest Size Percentile', 'image', 0, 100, nothing)
	cv2.createTrackbar('Highest Size Percentile', 'image', 99, 100, nothing)
	threshImg = img
	while(1):
		k = cv2.waitKey(1)
		if k == 32:
			break
		try:
			# threshImg = cv2.resize(threshImg, (1900, 1200))
			cv2.imshow('image', threshImg)
		except:
			print 'WARNING: cv2 did not read the image correctly'

		# get current positions of four trackbars
		lowerPercentile = cv2.getTrackbarPos('Lowest Size Percentile','image')
		higherPercentile = cv2.getTrackbarPos('Highest Size Percentile','image')

		if (lowerPercentile != sizeRange[0] or higherPercentile != sizeRange[1]):
			sizeRange[0] = lowerPercentile
			sizeRange[1] = higherPercentile
			threshImg = adjustSizeFilter(img, lowerPercentile, higherPercentile)


	cv2.destroyAllWindows()
	return sizeRange, threshImg

def adjustSizeFilter(img, lowerPercentile, higherPercentile):
	label_img, cc_num = nd.label(img)
	objs = nd.find_objects(label_img)
	areas = nd.sum(img, label_img, range(cc_num+1))

	indices = sorted(range(len(areas)), key = lambda k: areas[k])

	orderedAreas = [areas[ind] for ind in indices]

	lowerThresh = orderedAreas[int((float(lowerPercentile)/100) * len(orderedAreas))]
	if higherPercentile != 100:
		upperThresh = orderedAreas[int((float(higherPercentile)/100) * len(orderedAreas))]
	else:
		upperThresh = orderedAreas[-1]
	print lowerThresh
	print len(indices)
	# print len(orderedAreas)
	# print len(objs)
	# print img.shape

	area_mask = (areas < lowerThresh)
	label_img[area_mask[label_img]] = 0

	area_mask = (areas > upperThresh)
	label_img[area_mask[label_img]] = 0

	# print np.ndarray.dtype(label_img)
	label_img[np.where(label_img > 0)] = 2**16

	return label_img

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

def saveAndQuit(oldThresh, noiseKernel, sizeRange):
	print "Writing configuration file..."
	cfgfile = open("saar.ini",'w')
	Config = ConfigParser.ConfigParser()
	Config.add_section('Options')
	Config.set('Options','Threshold Value', oldThresh)
	Config.set('Options','Remove Noise Kernel Size', noiseKernel)
	Config.set('Options','Filter Size Lower Bound', sizeRange[0])
	Config.set('Options','Filter Size Upper Bound', sizeRange[1])
	Config.write(cfgfile)
	cfgfile.close()

def main():
	emFolderPath = sys.argv[1]
	emPaths = sorted(glob.glob(emFolderPath +'*.tif*'))

	em = emPaths[0]
	img = cv2.imread(em, 0)
	img = np.uint8(img)
	noiseKernel = 0

	while True:
		print("SAAR MENU")
		print("1. Threshold")
		print("2. Remove Noise")
		print("3. Filter by Size")
		print("4. Save and Quit")
		choice = raw_input(">")
		if choice=='1':
			oldThresh, threshImg = threshVis(img)
		elif choice=='2':
			noiseKernel, threshImg = noiseVis(threshImg)
			tifffile.imsave('junk/test.tif', threshImg)
		elif choice=='3':
			sizeRange, threshImg = sizeVis(threshImg)
		elif choice=='4':
			saveAndQuit(oldThresh, noiseKernel, sizeRange)
			break
		else:
			continue




if __name__ == "__main__":
	main()
