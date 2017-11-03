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
from marching_cubes import march
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
import matplotlib.pyplot as plt
import queue
import ast

# Needed for multithreading
NUMBERCORES = multiprocessing.cpu_count()
print("Found " + str(NUMBERCORES) + " number of cores. Using " + str(NUMBERCORES - 1) + ".")
NUMBERCORES -= 1

threshVals = []
pVals = []
lowerSizeVals = []
upperSizeVals = []
blobRecoveryRadii = []
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

def nothing(x):
    pass

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

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def modifiedConnectedComponents(massFolderPath, labelsFolderPath):

	threshPaths = sorted(glob.glob(massFolderPath +'*.tif*'))

	images = [cv2.imread(threshPaths[z], -1) for z in range(len(threshPaths))]

	ii = 0
	for value in chunks(images, 100):
		print(ii)
		print("loaded")
		images = np.dstack(value)
		print("stacked")

		label_img, number = nd.measurements.label(images)

		images = np.uint16(label_img)


		for each in range(images.shape[2]):
			print(each + ii)
			img = images[:,:,each]
			tifffile.imsave(labelsFolderPath + str(os.path.basename(threshPaths[each+ ii])), img)
		ii += images.shape[2]

def calcMesh(label, meshes):
	print(label)
	#code.interact(local=locals())
	indices = np.where(labelStack==label)
	box, dimensions = findBBDimensions(indices)
	print(box)
	if dimensions[0] > 500 or dimensions[1] > 500 or dimensions[2] > 500:
		print('skipped')
		return

	window = labelStack[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
	localIndices = np.where(window==label)
	blankImg = np.zeros(window.shape, dtype=bool)
	blankImg[localIndices] = 1
	try:
		vertices, normals, faces = march(blankImg.transpose(), 0)  # zero smoothing rounds
		print("success on mesh")
	except:
		print("failed")
		return
	print("writing")
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
	global labelStack
	labelStack = [tifffile.imread(labelsPaths[z]) for z in range(len(labelsPaths))]
	labelStack = np.dstack(labelStack)
	print("Loaded data...")

	for i, itm in enumerate(itemlist):
		calcMesh(itm, meshesFolderPath)
		end = timer()
		print(str(i+1) + "/" + str(len(itemlist)) + " time: " + str(end-start))

def trackSize(labelStack, axis, start, minLabelSize):
	# Track the sizes of labels in the stack along a particular axis
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
	# Makes a list of all the labels in the stack bigger than a given size, saves it to a numpy file to be used for meshing
	start = timer()
	labelsPaths = sorted(glob.glob(labelsFolderPath +'*.tif*'))
	labelStack = [tifffile.imread(labelsPaths[z]) for z in range(len(labelsPaths))]

	labelStack = np.dstack(labelStack)

	print("X Direction...")
	finalListX = trackSize(labelStack, 0, start, minLabelSize)
	print("Y Direction...")
	finalListY = trackSize(labelStack, 1, start, minLabelSize)
	print("Z Direction...")
	finalListZ = trackSize(labelStack, 2, start, minLabelSize)

	finalList = list(set(finalListX) | set(finalListY) | set(finalListZ))

	np.save('outfile.npy', finalList)

def dilateLabels(image):
	# Dilates all the labels in an image individually
	name = image[0]
	img = image[1]
	labelsFolderPath = image[2]
	threshPaths = image[3]
	new = np.zeros(img.shape, dtype=np.uint32)
	labels = np.unique(img)
	kernel = np.ones((3,3), np.uint8)
	for label in labels:
		pixels = np.where(img == label)
		if len(pixels[0]) < 5:
			continue
		blank = np.zeros(img.shape)
		blank[pixels] = 1
		blank = cv2.dilate(blank, kernel, iterations=2)
		pixels = np.where(blank == 1)
		new[pixels] = label

	tifffile.imsave(labelsFolderPath + str(os.path.basename(name)), new)

def connectedComponents(massFolderPath, labelsFolderPath):
	# Uses scipy's connected components function to label the individual axons. Applies an erosion first, then the labeling function, then dilates each label individually
	threshPaths = sorted(glob.glob(massFolderPath +'*.tif*'))

	kernel = np.ones((3,3),np.uint8)
	alreadyDone = glob.glob(labelsFolderPath + "*.tif*")
	images = [cv2.erode(cv2.imread(threshPaths[z], -1), kernel, 2) for z in range(len(threshPaths))]

	print("loaded")
	images = np.dstack(images)
	print("stacked")

	label_img, number = nd.measurements.label(images)

	images = np.uint32(label_img)

	images = [(threshPaths[z], images[:,:,z], labelsFolderPath, threshPaths) for z in range(len(threshPaths))]

	pool = ThreadPool(NUMBERCORES)
	for i, _ in enumerate(pool.imap_unordered(dilateLabels, images), 1):
		sys.stderr.write('\rdone {0:%}'.format(i/len(images)))

def adjustNoise(threshImg, ks):
	kernelImg = cv2.morphologyEx(threshImg, cv2.MORPH_OPEN, np.ones((ks,ks)))
	ret,kernelImg = cv2.threshold(kernelImg, 0, 255, cv2.THRESH_BINARY)

	return kernelImg

def processSlice(imgPath):
	# Process an individual image
	img = cv2.imread(imgPath[0], -1)
	img = np.uint8(img)

	outImg = adjustThresh(img, imgPath[1])
	outImg = adjustNoise(outImg, imgPath[2])
	outImg = adjustSizeFilterVis(outImg, imgPath[3], imgPath[4])

	tifffile.imsave(massFolderPath + str(os.path.basename(imgPath[0])), outImg)

	return outImg

def applyParams(emPaths):
	# Applies the parameters set in getParameters to the entire stack. Performs linear interpolation on the given parameters to get values for each individual slice

	emImages = [cv2.imread(path,-1) for path in emPaths]
	cfgfile = open("saar.ini",'r')
	config = configparser.ConfigParser()
	config.read('saar.ini')

	try:
		threshDict = ast.literal_eval(config.get('Options', 'Threshold Value'))
		threshInterp = np.interp(list(range(len(emPaths))),list(threshDict.keys()),list(threshDict.values()))
		global threshVals
		threshVals = [int(round(i)) for i in threshInterp]
	except:
		print("threshVal not found in config file, did you set the parameters?")
	try:
		pDict = ast.literal_eval(config.get('Options', 'Remove Noise Kernel Size'))
		pInterp = np.interp(list(range(len(emPaths))),list(pDict.keys()),list(pDict.values()))
		global pVals
		pVals = [int(round(i)) for i in pInterp]
	except:
		print("kernel value not found in config file, did you set the parameters?")
	try:
		sizeDict = ast.literal_eval(config.get('Options', 'Filter Size Range'))
		lowerSizeInterp = np.interp(list(range(len(emPaths))),list(sizeDict.keys()),[s[0] for s in list(sizeDict.values())])
		global lowerSizeVals
		lowerSizeVals = [int(round(i)) for i in lowerSizeInterp]
		upperSizeInterp = np.interp(list(range(len(emPaths))),list(sizeDict.keys()),[s[1] for s in list(sizeDict.values())])
		global upperSizeVals
		upperSizeVals = [int(round(i)) for i in upperSizeInterp]
	except:
		print("size values not found in config file, did you set the parameters?")
	try:
		recoveryDict = ast.literal_eval(config.get('Options', 'Blob Recovery Radius'))
		blobRecoveryInterp = np.interp(list(range(len(emPaths))),list(recoveryDict.keys()),list(recoveryDict.values()))
		global blobRecoveryRadii
		blobRecoveryRadii = [int(round(i)) for i in blobRecoveryInterp]
	except:
		print("recovery radius value not found in config file, did you set the parameters?")

	emPathsWithParameters = [[path, threshVals[i], pVals[i], lowerSizeVals[i], upperSizeVals[i], blobRecoveryRadii[i]] for i, path in enumerate(emPaths)]
	pool = ThreadPool(NUMBERCORES)

	for i, _ in enumerate(pool.imap_unordered(processSlice, emPathsWithParameters), 1):
		sys.stderr.write('\rdone {0:%}'.format(i/len(emPaths)))

	return emImages

def findCentroid(listofpixels):
	# Finds the centroid for a list of pixel coordinates
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

def adjustRecoveryRadius(labelImg, lowerAreaMask, upperAreaMask, r):
	# Brings back blobs that were eliminated by the size filter if they are part of an axon bundle, may also bring back some noise

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

def recoverVis(img, intactLabelImg, lowerAreaMask, upperAreaMask, z):
	# Trackbar allows user to set radius for blob recovery and view the results in real time, space bar confirms value
	oldRadius = 0
	r = oldRadius
	cv2.namedWindow('image at z=' + str(z))

	cv2.createTrackbar('Blob Recovery Radius', 'image at z=' + str(z), 0, 25, nothing)
	labelImg = intactLabelImg.copy()

	labelImg[lowerAreaMask[labelImg]] = 0

	labelImg[upperAreaMask[labelImg]] = 0

	labelImg[np.where(labelImg > 0)] = 2**16

	while(1):
		k = cv2.waitKey(1)
		if k == 32:
			break
		try:
			cv2.imshow('image at z=' + str(z), labelImg)
		except:
			print('WARNING: cv2 did not read the image correctly')

		r = cv2.getTrackbarPos('Blob Recovery Radius','image at z=' + str(z))

		if (r != oldRadius):
			oldRadius = r
			labelImg = adjustRecoveryRadius(intactLabelImg.copy(), lowerAreaMask.copy(), upperAreaMask, r)


	cv2.destroyAllWindows()
	return oldRadius, labelImg

def adjustSizeFilterVis(img, lowerPercentile, higherPercentile):
	# Orders blobs in an image by size and keeps only those within the given percentile range
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

	labelImg[np.where(labelImg > 0)] = 2**16

	return labelImg

def sizeVis(img, z):
	# Trackbar allows user to set range for size filter and view the results in real time, space bar confirms value
	sizeRange = [0,999]
	cv2.namedWindow('image at z=' + str(z))

	cv2.createTrackbar('Lowest Size Percentile', 'image at z=' + str(z), 0, 1000, nothing)
	cv2.createTrackbar('Highest Size Percentile', 'image at z=' + str(z), 999, 1000, nothing)
	threshImg = img

	while(1):
		k = cv2.waitKey(1)
		if k == 32:
			break
		try:
			cv2.imshow('image at z=' + str(z), threshImg)
		except:
			print('WARNING: cv2 did not read the image correctly')

		lowerPercentile = cv2.getTrackbarPos('Lowest Size Percentile','image at z=' + str(z))
		higherPercentile = cv2.getTrackbarPos('Highest Size Percentile','image at z=' + str(z))

		if (lowerPercentile != sizeRange[0] or higherPercentile != sizeRange[1]):
			sizeRange[0] = lowerPercentile
			sizeRange[1] = higherPercentile
			threshImg, intactLabelImg, lowerAreaMask, upperAreaMask = adjustSizeFilterVis(img, lowerPercentile, higherPercentile)


	cv2.destroyAllWindows()
	return sizeRange, threshImg, intactLabelImg, lowerAreaMask, upperAreaMask

def noiseVis(threshImg, z):
	# Trackbar allows user to set kernel size for noise removal and view the results in real time, space bar confirms value
	oldKernel = 2
	ks = oldKernel
	cv2.namedWindow('image at z=' + str(z))
	threshImg = np.uint8(threshImg)

	kernelImg = np.uint8(threshImg)
	cv2.createTrackbar('Kernel Size for Noise Removal', 'image at z=' + str(z), 1, 10, nothing)
	while(1):
		k = cv2.waitKey(1)
		if k == 32:
			break
		try:
			cv2.imshow('image at z=' + str(z), kernelImg)

		except:
			print('WARNING: cv2 did not read the image correctly')
		ks = cv2.getTrackbarPos('Kernel Size for Noise Removal', 'image at z=' + str(z))

		if (ks != oldKernel):
			oldKernel = ks
			kernelImg = cv2.morphologyEx(threshImg, cv2.MORPH_OPEN, np.ones((ks,ks)))
			ret,kernelImg = cv2.threshold(kernelImg, 0, 255, cv2.THRESH_BINARY)

	cv2.destroyAllWindows()
	return oldKernel, kernelImg

def adjustThresh(originalImg, globalValue):
	# Performs a global threshold, dilates the image, fills holes, erodes, and thresholds again to make sure the image is binary
	ret,thresh1 = cv2.threshold(originalImg, int(globalValue), 255, cv2.THRESH_BINARY)
	kernel = np.ones((3,3),np.uint8)
	thresh1 = cv2.dilate(thresh1, kernel, 1)
	thresh1 = np.uint8(nd.morphology.binary_fill_holes(thresh1))
	ret,thresh1 = cv2.threshold(thresh1, 0, 255, cv2.THRESH_BINARY)
	thresh1 = cv2.erode(thresh1, kernel, 1)

	return thresh1

def threshVis(img, z):
	# Trackbar allows user to set threshold value and view the results in real time, space bar confirms value
	oldThresh = 200
	cv2.namedWindow('image at z=' + str(z))

	cv2.createTrackbar('Global Threshold', 'image at z=' + str(z), 0, 255, nothing)
	threshImg = img
	while(1):
		k = cv2.waitKey(1)
		if k == 32:
			break
		try:
			cv2.imshow('image at z=' + str(z), threshImg)
		except:
			print('WARNING: cv2 did not read the image correctly')

		g = cv2.getTrackbarPos('Global Threshold','image at z=' + str(z))
		if (g != oldThresh):
			oldThresh = g
			threshImg = adjustThresh(img.copy(), g)

	cv2.destroyAllWindows()

	return oldThresh, threshImg

def getParameters(emPaths, sampleIndices):
	# Get parameters from the user for each of the slices to be sampled and write them to an ini file
	sampleImgs = []
	for index in sampleIndices:
		sampleImgs.append(np.uint8(cv2.imread(emPaths[index], -1)))

	threshDict = {}
	noiseDict = {}
	sizeDict = {}
	recoveryDict = {}
	for i, img in enumerate(sampleImgs):
		oldThresh, threshImg = threshVis(img, sampleIndices[i])

		noiseKernel, threshImg = noiseVis(threshImg, sampleIndices[i])

		sizeRange, threshImg, intactLabelImg, lowerAreaMask, upperAreaMask = sizeVis(threshImg, sampleIndices[i])

		blobRecoveryRadius, threshImg = recoverVis(threshImg, intactLabelImg, lowerAreaMask, upperAreaMask, sampleIndices[i])

		threshDict[sampleIndices[i]] = oldThresh
		noiseDict[sampleIndices[i]] = noiseKernel
		sizeDict[sampleIndices[i]] = sizeRange
		recoveryDict[sampleIndices[i]] = blobRecoveryRadius

	print("Writing configuration file...")
	cfgfile = open("saar.ini",'w')
	Config = configparser.ConfigParser()
	Config.add_section('Options')
	Config.set('Options','Threshold Value', str(threshDict))
	Config.set('Options','Remove Noise Kernel Size', str(noiseDict))
	Config.set('Options','Filter Size Range', str(sizeDict))
	Config.set('Options','Blob Recovery Radius', str(recoveryDict))
	Config.write(cfgfile)
	cfgfile.close()

def getSampleIndices(n,numSamples):
	numPartitions = numSamples - 1
	increment = int(round(n/numPartitions))
	indices = [increment*a for a in range(numSamples)]
	indices[-1] = n
	return indices

def main():
	# This set of parameters yielded the best results so far:
	# threshold value = {0: 190, 915: 190, 1830: 190}
	# remove noise kernel size = {0: 3, 915: 3, 1830: 3}
	# filter size range = {0: [205, 993], 915: [180, 994], 1830: [190, 995]}
	# blob recovery radius = {0: 0, 915: 0, 1830: 0}
	# blob recovery wasn't used at all

	emFolderPath = sys.argv[1]
	emPaths = sorted(glob.glob(emFolderPath +'*.tif*'))

	global massFolderPath
	massFolderPath = sys.argv[2]

	labelsFolderPath = sys.argv[3]

	global meshesFolderPath
	meshesFolderPath = sys.argv[4]

	# Number of slices presented to the user in getParameters
	numberOfSamples = 3

	sampleIndices = getSampleIndices(len(emPaths)-1,numberOfSamples)

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
			getParameters(emPaths, sampleIndices)
		elif choice=='2':
			print("Enter a minimum label size:")
			# Any label smaller than this size will be eliminated in makeItemList
			# Recommended value: 150
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
