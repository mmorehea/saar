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
#from marching_cubes import march
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
blobRecoveryRadius = 0
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
	# threshImg = cv2.resize(threshImg, (950*2, 750*2))
	while(1):
		k = cv2.waitKey(1)
		if k == 32:
			break
		try:
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
	# kernelImg = cv2.resize(kernelImg, (950*2, 750*2))
	cv2.createTrackbar('Kernel Size for Noise Removal', 'image', 1, 10, nothing)
	while(1):
		k = cv2.waitKey(1)
		if k == 32:
			break
		try:
			cv2.imshow('image', kernelImg)

		except:
			print('WARNING: cv2 did not read the image correctly')
		ks = cv2.getTrackbarPos('Kernel Size for Noise Removal', 'image')
		# get current positions of four trackbars

		if (ks != oldKernel):
			# print ks
			oldKernel = ks
			kernelImg = cv2.morphologyEx(threshImg, cv2.MORPH_OPEN, np.ones((ks,ks)))
			# kernelImg = cv2.dilate(kernelImg, (2,2), iterations=ks)
			ret,kernelImg = cv2.threshold(kernelImg, 0, 255, cv2.THRESH_BINARY)

	cv2.destroyAllWindows()
	return oldKernel, kernelImg

def adjustRecoveryRadius(intactLabelImg, lowerAreaMask, upperAreaMask, r):
	labelImg = intactLabelImg.copy()
	r = r * 4 # Scaling up, this is so the user can change the radius in increments of 4
	# Remove small axons within bundles from the area mask
	# r is the maximum distance for a blob to be considered a neighbor
	minNeighborCount = 5 # minimum number of neighbors to remove blob from area mask
	for i, value in enumerate(lowerAreaMask):
		if value == True:
			a = np.where(labelImg==i)
			label = list(zip(a[0],a[1]))


			centroid = findCentroid(label)

			y,x = np.ogrid[-centroid[0]:labelImg.shape[0]-centroid[0], -centroid[1]:labelImg.shape[1]-centroid[1]]
			mask = x*x + y*y <= r*r

			neighborLabels = [lab for lab in np.unique(labelImg[mask]) if lab > 0 and lab != labelImg[label[0]]]

			if len(neighborLabels) > minNeighborCount:
				lowerAreaMask[i] = False

	labelImg[lowerAreaMask[labelImg]] = 0


	labelImg[upperAreaMask[labelImg]] = 0

	labelImg[np.where(labelImg > 0)] = 2**16

	return labelImg

def recoverVis(img, intactLabelImg, lowerAreaMask, upperAreaMask):

	oldRadius = 0
	r = oldRadius
	cv2.namedWindow('image')

	cv2.createTrackbar('Blob Recovery Radius', 'image', 0, 25, nothing)
	labelImg = intactLabelImg.copy()

	labelImg[lowerAreaMask[labelImg]] = 0

	labelImg[upperAreaMask[labelImg]] = 0

	labelImg[np.where(labelImg > 0)] = 2**16

	# threshImg = cv2.resize(threshImg, (950*2, 750*2))
	# ret,threshImg = cv2.threshold(threshImg, 0, 255, cv2.THRESH_BINARY)
	while(1):
		k = cv2.waitKey(1)
		if k == 32:
			break
		try:
			cv2.imshow('image', labelImg)
		except:
			print('WARNING: cv2 did not read the image correctly')

		# get current positions of four trackbars
		r = cv2.getTrackbarPos('Blob Recovery Radius','image')

		if (r != oldRadius):
			oldRadius = r
			labelImg = adjustRecoveryRadius(intactLabelImg, lowerAreaMask, upperAreaMask, r)


	cv2.destroyAllWindows()
	return oldRadius, labelImg

def sizeVis(img):

	sizeRange = [0,999]
	cv2.namedWindow('image')

	cv2.createTrackbar('Lowest Size Percentile', 'image', 0, 1000, nothing)
	cv2.createTrackbar('Highest Size Percentile', 'image', 999, 1000, nothing)
	threshImg = img
	# threshImg = cv2.resize(threshImg, (950*2, 750*2))
	# ret,threshImg = cv2.threshold(threshImg, 0, 255, cv2.THRESH_BINARY)
	while(1):
		k = cv2.waitKey(1)
		if k == 32:
			break
		try:
			cv2.imshow('image', threshImg)
		except:
			print('WARNING: cv2 did not read the image correctly')

		# get current positions of four trackbars
		lowerPercentile = cv2.getTrackbarPos('Lowest Size Percentile','image')
		higherPercentile = cv2.getTrackbarPos('Highest Size Percentile','image')

		if (lowerPercentile != sizeRange[0] or higherPercentile != sizeRange[1]):
			sizeRange[0] = lowerPercentile
			sizeRange[1] = higherPercentile
			threshImg, intactLabelImg, lowerAreaMask, upperAreaMask = adjustSizeFilterVis(img, lowerPercentile, higherPercentile)


	cv2.destroyAllWindows()
	return sizeRange, threshImg, intactLabelImg, lowerAreaMask, upperAreaMask

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
	labelImg, cc_num = nd.label(img)
	objs = nd.find_objects(labelImg)
	areas = nd.sum(img, labelImg, range(cc_num+1))

	indices = sorted(range(len(areas)), key = lambda k: areas[k])

	orderedAreas = [areas[ind] for ind in indices]

	lowerThresh = orderedAreas[int((float(lowerPercentile)/1000) * len(orderedAreas))]
	if higherPercentile != 1000:
		upperThresh = orderedAreas[int((float(higherPercentile)/1000) * len(orderedAreas))]
	else:
		upperThresh = orderedAreas[-1]

	intactLabelImg = labelImg.copy()

	lowerAreaMask = (areas < lowerThresh)
	lowerAreaMask[0] = False
	labelImg[lowerAreaMask[labelImg]] = 0

	upperAreaMask = (areas > upperThresh)
	labelImg[upperAreaMask[labelImg]] = 0

	# print np.ndarray.dtype(label_img)
	labelImg[np.where(labelImg > 0)] = 2**16

	tifffile.imsave('labelImg', labelImg)
	tifffile.imsave('intactLabelImg', intactLabelImg)

	return labelImg, intactLabelImg, lowerAreaMask, upperAreaMask

def adjustSizeFilter(img, lowerPercentile, higherPercentile, blobRecoveryRadius):
	labelImg, cc_num = nd.label(img)
	objs = nd.find_objects(labelImg)
	areas = nd.sum(img, labelImg, range(cc_num+1))

	indices = sorted(range(len(areas)), key = lambda k: areas[k])

	orderedAreas = [areas[ind] for ind in indices]

	lowerThresh = orderedAreas[int((float(lowerPercentile)/1000) * len(orderedAreas))]
	if higherPercentile != 1000:
		upperThresh = orderedAreas[int((float(higherPercentile)/1000) * len(orderedAreas))]
	else:
		upperThresh = orderedAreas[-1]

	areaMask = (areas < lowerThresh)
	areaMask[0] = False

	# Remove small axons within bundles from the area mask
	r = blobRecoveryRadius * 4 # maximum distance for a blob to be considered a neighbor
	minNeighborCount = 5 # minimum number of neighbors to remove blob from area mask
	for i, value in enumerate(areaMask):
		if value == True:
			a = np.where(labelImg==i)
			label = list(zip(a[0],a[1]))


			centroid = findCentroid(label)

			y,x = np.ogrid[-centroid[0]:labelImg.shape[0]-centroid[0], -centroid[1]:labelImg.shape[1]-centroid[1]]
			mask = x*x + y*y <= r*r

			neighborLabels = [lab for lab in np.unique(labelImg[mask]) if lab > 0 and lab != labelImg[label[0]]]

			if len(neighborLabels) > minNeighborCount:
				areaMask[i] = False

	labelImg[areaMask[labelImg]] = 0


	areaMask = (areas > upperThresh)
	labelImg[areaMask[labelImg]] = 0

	# print np.ndarray.dtype(labelImg)
	labelImg[np.where(labelImg > 0)] = 2**16

	return labelImg

def adjustNoise(threshImg, ks):
	kernelImg = cv2.morphologyEx(threshImg, cv2.MORPH_OPEN, np.ones((ks,ks)))
	ret,kernelImg = cv2.threshold(kernelImg, 0, 255, cv2.THRESH_BINARY)
	# kernelImg = cv2.erode(kernelImg, (ks,ks), iterations = 6)

	return kernelImg


def processSlice(imgPath):
	img = cv2.imread(imgPath, -1)
	img = np.uint8(img)
	# img = cv2.bitwise_not(img)

	outImg = adjustThresh(img, threshVal)

	outImg = adjustNoise(outImg, p)

	outImg = adjustSizeFilter(outImg, lowerSizeVal, upperSizeVal, blobRecoveryRadius)

	tifffile.imsave(massFolderPath + str(os.path.basename(imgPath)), outImg)

	return outImg

def getParameters(img):
	# img = cv2.bitwise_not(img)
	oldThresh, threshImg = threshVis(img)
	# tifffile.imsave('afterthresh2.tif', threshImg)

	noiseKernel, threshImg = noiseVis(threshImg)
	# tifffile.imsave('afternoise2.tif', threshImg)

	sizeRange, threshImg, intactLabelImg, lowerAreaMask, upperAreaMask = sizeVis(threshImg)
	# tifffile.imsave('threshImg.tif', threshImg)

	blobRecoveryRadius, threshImg = recoverVis(threshImg, intactLabelImg, lowerAreaMask, upperAreaMask)


	print("Writing configuration file...")
	cfgfile = open("saar.ini",'w')
	Config = configparser.ConfigParser()
	Config.add_section('Options')
	Config.set('Options','Threshold Value', str(oldThresh))
	Config.set('Options','Remove Noise Kernel Size', str(noiseKernel))
	Config.set('Options','Filter Size Lower Bound', str(sizeRange[0]))
	Config.set('Options','Filter Size Upper Bound', str(sizeRange[1]))
	Config.set('Options','Blob Recovery Radius', str(blobRecoveryRadius))
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
	try:
		global blobRecoveryRadius
		blobRecoveryRadius = int(config.get('Options', 'Blob Recovery Radius'))

	except:
		print("recovery radius value not found in config file, did you set the parameters?")

	pool = ThreadPool(NUMBERCORES)
	#
	for i, _ in enumerate(pool.imap_unordered(processSlice, emPaths), 1):
		sys.stderr.write('\rdone {0:%}'.format(i/len(emPaths)))

	# for each in emPaths:
	# 	processSlice(each)

	# processedStack = pool.map(processSlice, images)

	# print "time, mass, single: " + str(timer() - start)
	return emImages

def connectedComponents(massFolderPath, labelsFolderPath):

	threshPaths = sorted(glob.glob(massFolderPath +'*.tif*'))

	images = [cv2.imread(threshPaths[z], -1) for z in range(len(threshPaths))]
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
	for i in range(labelStack.shape[axis]):
		end = timer()
		print(str(i) + "/" + str(labelStack.shape[axis]) + " time: " + str(end-start))

		if axis == 0:
			img = labelStack[i,:,:]
		elif axis == 1:
			img = labelStack[:,i,:]
		elif axis == 2:
			img = labelStack[:,:,i]

		idList = np.unique(img)


		for each in list(tracker):
			tracker[each][2] += 1
			if tracker[each][2] > 25:
				if tracker[each][1] - tracker[each][0] < minLabelSize:
					tracker.pop(each)

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

def calcMesh(label, meshes):
	print(label)

	indices = np.where(labelStack==label)
	box, dimensions = findBBDimensions(indices)
	print(box)
	if dimensions[0] > 500 and dimensions[1] > 500 and dimensions[2] > 500:
		print('skipped')
		return

	window = labelStack[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
	localIndices = np.where(window==label)
	blankImg = np.zeros(window.shape, dtype=bool)
	blankImg[localIndices] = 1
	try:
		vertices, normals, faces = march(blankImg.transpose(), 1)  # zero smoothing rounds
	except:
		return

	with open(meshes + str(label)+".obj", 'w') as f:
		f.write("# OBJ file\n")
		for v in vertices:
			f.write("v %.2f %.2f %.2f \n" % ((box[0] * SCALEX) + (v[2] * SCALEX) + XOFFSET, (box[2] * SCALEY) + (v[1] * SCALEY) + YOFFSET, (box[4] * SCALEZ) + v[0] * 5.454545))
		for n in normals:
			f.write("vn %.2f %.2f %.2f \n" % (n[2], n[1], n[0]))
		for face in faces:
			f.write("f %d %d %d \n" % (face[0]+1, face[1]+1, face[2]+1))

def generateMeshes(meshesFolderPath, labelsFolderPath):
	start = timer()
	q = queue.Queue()

	alreadyDone = glob.glob(meshesFolderPath + "*.obj")

	alreadyDone = sorted([int(os.path.basename(i)[:-4]) for i in alreadyDone])
	print(alreadyDone)

	with open ('outfile.npy', 'rb') as fp:
		itemlist = np.load(fp)
		itemlist = itemlist[10:] # Why is this?

	itemlist = sorted([itm for itm in itemlist if itm not in alreadyDone])

	print("Found labels...")
	print("firstlabel: " + str(itemlist[0]))
	print("Number of labels", str(len(itemlist)))

	labelsPaths = sorted(glob.glob(labelsFolderPath +'*.tif*'))
	#code.interact(local=locals())
	global labelStack
	labelStack = [tifffile.imread(labelsPaths[z]) for z in range(len(labelsPaths))]
	labelStack = np.dstack(labelStack)
	print("Loaded data...")

	for i, itm in enumerate(itemlist):
		calcMesh(itm, meshes)
		end = timer()
		print(str(i+1) + "/" + str(len(itemlist)) + " time: " + str(end-start))

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

	emImg = img.copy()

	th3 = cv2.adaptiveThreshold(emImg.copy(),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,191,2)
	labelImg, cc_num = nd.label(th3)
	tifffile.imsave('xxxxxxxxx', labelImg)

	areas = nd.sum(th3, labelImg, range(cc_num+1))

	indices = sorted(range(len(areas)), key = lambda k: areas[k])

	orderedAreas = [areas[ind] for ind in indices]

	lowerAreaMask = (areas < 650)
	lowerAreaMask[0] = False
	labelImg[lowerAreaMask[labelImg]] = 0
	# cv2.imshow('l',labelImg)
	# cv2.waitKey()
	tifffile.imsave('yyyyyyyyy', labelImg)
	code.interact(local=locals())
	for label in range(cc_num+1)[1:]:
		b = np.where(labelImg == label)

		blank = np.zeros(labelImg.shape, np.uint8)
		blank[b] = 99999

		# im, contours,hierarchy = cv2.findContours(blank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



	while True:
		print("SAAR MENU")
		print("1. Set Parameters")
		print("2. Run 3-5")
		print("\t3. Apply Parameters to Whole Stack")
		print("\t4. Connected Components Labeling")
		print("\t5. Filter Labels by Size (for easier meshing)")
		print("6. Generate Meshes (use vol2mesh for now)")
		print("7. Separate False Merges (not ready yet)")
		print("8. Quit")
		choice = input(">")
		if choice=='1':
			getParameters(img)
		elif choice=='2':
			print("Enter a minimum label size:")
			minLabelSize = int(input(">"))
			emImages = applyParams(emPaths)
			connectedComponents(massFolderPath, labelsFolderPath)
			makeItemList(labelsFolderPath, minLabelSize)
		elif choice=='3':
			emImages = applyParams(emPaths)
		elif choice=='4':
			connectedComponents(massFolderPath, labelsFolderPath)
		elif choice=='5':
			print("Enter a minimum label size:")
			minLabelSize = int(input(">"))
			makeItemList(labelsFolderPath, minLabelSize)
		elif choice=='6':
			generateMeshes(meshesFolderPath, labelsFolderPath)
		elif choice=='7':
			continue
		elif choice=='8':
			sys.exit()
		else:
			continue




if __name__ == "__main__":
	main()
