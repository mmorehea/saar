from __future__ import division
import cv2
import numpy as np
from numpy import load
import os
import sys
import glob
import code
import tifffile
import threading

from timeit import default_timer as timer
import configparser
from scipy import ndimage as nd
from skimage import measure
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
from itertools import cycle
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import threading
import pickle
import math

# from mass.py
NUMBERCORES = multiprocessing.cpu_count()
print("Found " + str(NUMBERCORES) + " number of cores. Using " + str(NUMBERCORES - 1) + ".")
NUMBERCORES -= 1

threshVal = 0
p = 0
lowerSizeVal = 0
upperSizeVal = 0
massFolderPath = ''
# -----

# from multiMesh3.py
SCALEX = 10.0
SCALEY = 10.0
SCALEZ = 1.0

XOFFSET = 456.0
YOFFSET = 456.0
labelStack = []
meshesFolderPath = []
# -----

def findBBDimensions(listOfPixels):
	xs = listOfPixels[0]
	ys = listOfPixels[1]
	zs = listOfPixels[2]

	minxs = min(xs)
	maxxs = max(xs)

	minys = min(ys)
	maxys = max(ys)

	minzs = min(zs)
	maxzs = max(zs)

	dx = maxxs - minxs
	dy = maxys - minys
	dz = maxzs - minzs

	return [minxs-2, maxxs+2, minys-2, maxys+2, minzs-2, maxzs+2], [dx, dy, dz]

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
			threshImg = cv2.resize(threshImg, (950, 750))
			cv2.imshow('image', threshImg)
		except:
			print('WARNING: cv2 did not read the image correctly')

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
			kernelImg = cv2.resize(kernelImg, (950, 750))
			cv2.imshow('image', kernelImg)
		except:
			print('WARNING: cv2 did not read the image correctly')
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
			threshImg = cv2.resize(threshImg, (950, 750))
			cv2.imshow('image', threshImg)
		except:
			print('WARNING: cv2 did not read the image correctly')

		# get current positions of four trackbars
		lowerPercentile = cv2.getTrackbarPos('Lowest Size Percentile','image')
		higherPercentile = cv2.getTrackbarPos('Highest Size Percentile','image')

		if (lowerPercentile != sizeRange[0] or higherPercentile != sizeRange[1]):
			sizeRange[0] = lowerPercentile
			sizeRange[1] = higherPercentile
			threshImg = adjustSizeFilterVis(img, lowerPercentile, higherPercentile)


	cv2.destroyAllWindows()
	return sizeRange, threshImg

def findCentroid(listofpixels):
	if len(listofpixels) == 0:
		return (0,0)
	rows = [p[0] for p in listofpixels]
	cols = [p[1] for p in listofpixels]
	try:
		centroid = int(round(np.mean(rows))), int(round(np.mean(cols)))
	except:
		print('error')
		code.interact(local=locals())
		centroid = (0,0)
	return centroid

def adjustSizeFilterVis(img, lowerPercentile, higherPercentile):
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

	area_mask = (areas < lowerThresh)
	label_img[area_mask[label_img]] = 0

	area_mask = (areas > upperThresh)
	label_img[area_mask[label_img]] = 0

	# print np.ndarray.dtype(label_img)
	label_img[np.where(label_img > 0)] = 2**16

	return label_img

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

	area_mask = (areas < lowerThresh)
	area_mask[0] = False

	# Remove small axons within bundles from the area mask
	r = 20 # maximum distance for a blob to be considered a neighbor
	minNeighborCount = 5 # minimum number of neighbors to remove blob from area mask
	for i, value in enumerate(area_mask):
		if value == True:
			a = np.where(label_img==i)
			label = zip(a[0],a[1])


			centroid = findCentroid(label)

			y,x = np.ogrid[-centroid[0]:label_img.shape[0]-centroid[0], -centroid[1]:label_img.shape[1]-centroid[1]]
			mask = x*x + y*y <= r*r

			neighborLabels = [lab for lab in np.unique(label_img[mask]) if lab > 0 and lab != label_img[zip(*label)][0]]

			if len(neighborLabels) > minNeighborCount:
				area_mask[i] = False

	label_img[area_mask[label_img]] = 0


	area_mask = (areas > upperThresh)
	label_img[area_mask[label_img]] = 0

	# print np.ndarray.dtype(label_img)
	label_img[np.where(label_img > 0)] = 2**16

	return label_img

def adjustNoise(threshImg, ks):
	kernelImg = cv2.morphologyEx(threshImg, cv2.MORPH_OPEN, np.ones((ks,ks)))
	ret,kernelImg = cv2.threshold(kernelImg, 0, 255, cv2.THRESH_BINARY)
	# kernelImg = cv2.erode(kernelImg, (ks,ks), iterations = 6)

	return kernelImg

def processSlice(imgPath):
	img = cv2.imread(imgPath, -1)
	img = np.uint8(img)

	outImg = adjustThresh(img, threshVal)

	outImg = adjustNoise(outImg, p)

	outImg = adjustSizeFilter(outImg, lowerSizeVal, upperSizeVal)

	tifffile.imsave(massFolderPath + str(os.path.basename(imgPath)), outImg)

	return outImg

def getParameters(img):
	oldThresh, threshImg = threshVis(img)
	noiseKernel, threshImg = noiseVis(threshImg)
	sizeRange, threshImg = sizeVis(threshImg)

	print("Writing configuration file...")
	cfgfile = open("saar.ini",'w')
	Config = configparser.ConfigParser()
	Config.add_section('Options')
	Config.set('Options','Threshold Value', oldThresh)
	Config.set('Options','Remove Noise Kernel Size', noiseKernel)
	Config.set('Options','Filter Size Lower Bound', sizeRange[0])
	Config.set('Options','Filter Size Upper Bound', sizeRange[1])
	Config.write(cfgfile)
	cfgfile.close()

def applyParams(emPaths):
	emImages = [cv2.imread(path,-1) for path in emPaths]
	cfgfile = open("saar.ini",'r')
	config = configparser.ConfigParser()
	config.read('saar.ini')

	try:
		global threshVal
		threshVal = int(config.get('Options', 'Threshold Value'))
	except:
		print("threshVal not found in config file, did you set the parameters?")
	try:
		global p
		p = int(config.get('Options', 'Remove Noise Kernel Size'))
	except:
		print("kernel value not found in config file, did you set the parameters?")
	try:
		global lowerSizeVal
		lowerSizeVal = int(config.get('Options', 'Filter Size Lower Bound'))
		global upperSizeVal
		upperSizeVal = int(config.get('Options', 'Filter Size Upper Bound'))
	except:
		print("size values not found in config file, did you set the parameters?")

	pool = ThreadPool(NUMBERCORES)

	for i, _ in enumerate(pool.imap_unordered(processSlice, emPaths), 1):
	    sys.stderr.write('\rdone {0:%}'.format(i/len(emPaths)))

	# processedStack = pool.map(processSlice, images)

	# print "time, mass, single: " + str(timer() - start)
	return emImages

def connectedComponents(massFolderPath, labelsFolderPath):

	threshPaths = sorted(glob.glob(massFolderPath +'*.tif*'))

	images = [cv2.imread(threshPaths[z], -1) for z in xrange(len(threshPaths))]
	print("loaded")
	images = np.dstack(images)
	print("stacked")


	label_img, number = nd.measurements.label(images)

	images = np.uint32(label_img)


	for each in range(images.shape[2]):
		print(each)
		img = images[:,:,each]
		tifffile.imsave(labelsFolderPath + str(os.path.basename(threshPaths[each])), img)

def trackSize(labelStack, axis, start, minLabelSize):
	tracker = {}
	for i in xrange(labelStack.shape[axis]):
		end = timer()
		print(str(i) + "/" + str(labelStack.shape[axis]) + " time: " + str(end-start))

		if axis == 0:
			img = labelStack[i,:,:]
		elif axis == 1:
			img = labelStack[:,i,:]
		elif axis == 2:
			img = labelStack[:,:,i]

		idList = np.unique(img)

		for each in tracker.keys():
			tracker[each][2] += 1
			if tracker[each][2] > 25:
				if tracker[each][1] - tracker[each][0] < minLabelSize:
					del tracker[each]

		for itm in idList:

			if itm not in tracker.keys():
				tracker[itm] = [i, 0, 0]
			else:
				if i > tracker[itm][1]:
					tracker[itm][1] = i + 1
					tracker[itm][2] = 0

	finalList = [t for t in tracker.keys() if tracker[t][1] - tracker[t][0] > minLabelSize]

	return finalList

def makeItemList(labelsFolderPath, minLabelSize):
	start = timer()
	labelsPaths = sorted(glob.glob(labelsFolderPath +'*.tif*'))
	labelStack = [tifffile.imread(labelsPaths[z]) for z in range(len(labelsPaths))]

	labelStack = np.dstack(labelStack)

	print("Loaded data... time: " + str(timer()-start))


	print("X Direction...")
	finalListX = trackSize(labelStack, 0, start, minLabelSize)
	print("Y Direction...")
	finalListY = trackSize(labelStack, 1, start, minLabelSize)
	print("Z Direction...")
	finalListZ = trackSize(labelStack, 2, start, minLabelSize)

	finalList = list(set(finalListX) | set(finalListY) | set(finalListZ))
	print(timer()-start)

	np.save('outfile.npy', finalList)


def main():
	start = timer()
	emFolderPath = sys.argv[1]
	emPaths = sorted(glob.glob(emFolderPath +'*.tif*'))

	global massFolderPath
	massFolderPath = sys.argv[2]
	labelsFolderPath = sys.argv[3]
	global meshesFolderPath
	meshesFolderPath = sys.argv[4]

	em = emPaths[0]
	img = cv2.imread(em, 0)
	img = np.uint8(img)

	while True:
		print("SAAR MENU")
		print("1. Set Parameters")
		print("2. Generate Meshes (takes about a day)")
		print("3. Separate False Merges")
		print("4. Quit")
		choice = input(">")
		if choice=='1':
			getParameters(img)
		elif choice=='2':
			print("Enter a minimum label size:")
			minLabelSize = int(raw_input(">"))
			emImages = applyParams(emPaths)
			connectedComponents(massFolderPath, labelsFolderPath)
			makeItemList(labelsFolderPath, minLabelSize)
			code.interact(local=locals())
		elif choice=='3':
			continue
		elif choice=='4':
			sys.exit()
		else:
			continue




if __name__ == "__main__":
	main()
