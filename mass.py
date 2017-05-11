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

from skimage import measure
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
from itertools import cycle

def adjustThresh(originalImg, value):
	ret,thresh1 = cv2.threshold(originalImg, int(value), 255, cv2.THRESH_BINARY)
	kernel = np.ones((3,3),np.uint8)
	thresh1 = cv2.dilate(thresh1, kernel, 1)
	thresh1 = np.uint8(nd.morphology.binary_fill_holes(thresh1))
	ret,thresh1 = cv2.threshold(thresh1, 0, 255, cv2.THRESH_BINARY)
	thresh1 = cv2.erode(thresh1, kernel, 1)
	return thresh1

def findCentroid(listofpixels):
	if len(listofpixels) == 0:
		return (0,0)
	rows = [p[0] for p in listofpixels]
	cols = [p[1] for p in listofpixels]
	try:
		centroid = int(round(np.mean(rows))), int(round(np.mean(cols)))
	except:
		print 'error'
		code.interact(local=locals())
		centroid = (0,0)
	return centroid

# Outdated function, prob do not need
def findNearestNeighbors(img, startPoint, oglabel):
	directions = cycle([[0,1], [0.5, 1], [1,1], [1,0.5], [1,0], [1,-0.5], [1,-1], [0.5,-1], [0,-1], [-0.5,-1], [-1,-1], [-1,-0.5], [-1,0], [-1,0.5], [-1,1], [-0.5,1]])
	increment = 0
	cycleCounter = 0
	distance = [0,0]
	neighbors = []

	while True:
		direction = directions.next()

		for i in [0,1]:
			if direction[i] > 0:
				distance[i] = direction[i] + increment
			elif direction[i] < 0:
				distance[i] = direction[i] - increment
			else:
				distance[i] = direction[i]

		checkPoint = [startPoint[0] + distance[0],startPoint[1] + distance[1]]

		if checkPoint[0] < 0:
			checkPoint[0] = 0
		elif checkPoint[0] > img.shape[0]-1:
			checkPoint[0] = img.shape[0]-1
		if checkPoint[1] < 0:
			checkPoint[1] = 0
		elif checkPoint[1] > img.shape[1]-1:
			checkPoint[1] = img.shape[1]-1

		checkPoint = tuple(checkPoint)

		# if increment == 19: code.interact(local=locals())

		cycleCounter += 1
		if cycleCounter % 8 == 0:
			increment += 1

		# print cycleCounter

		if img[checkPoint] > 0 and img[checkPoint] != oglabel and img[checkPoint] not in neighbors:
			neighbors.append(img[checkPoint])

		if increment > 25:
			break

	return neighbors

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

	# testing section
	# r=20
	# b = np.where(label_img==211)
	# testblob = zip(b[0],b[1])
	# centroid = findCentroid(testblob)
	# y,x = np.ogrid[-centroid[0]:label_img.shape[0]-centroid[0], -centroid[1]:label_img.shape[1]-centroid[1]]
	# mask = x*x + y*y <= r*r
	# neighborLabels = [lab for lab in np.unique(label_img[mask]) if lab > 0 and lab != label_img[zip(*testblob)][0]]
	# z = np.zeros(area_mask.shape, dtype=bool)
	# for l in neighborLabels:
	# 	z[l] = True
	# label_img[z[label_img]] = 99999
	# cv2.imshow('a',label_img)
	# cv2.waitKey()
	# code.interact(local=locals())

	# Remove small axons within bundles from the area mask
	r = 20 # minimum distance for a blob to be considered a neighbor
	for i, value in enumerate(area_mask):
		if value == True:
			a = np.where(label_img==i)
			label = zip(a[0],a[1])


			centroid = findCentroid(label)

			y,x = np.ogrid[-centroid[0]:label_img.shape[0]-centroid[0], -centroid[1]:label_img.shape[1]-centroid[1]]
			mask = x*x + y*y <= r*r

			neighborLabels = [lab for lab in np.unique(label_img[mask]) if lab > 0 and lab != label_img[zip(*label)][0]]
			# neighborLabels = findNearestNeighbors(label_img, centroid, label_img[zip(*label)][0])

			if len(neighborLabels) > 5:
				area_mask[i] = False

	label_img[area_mask[label_img]] = 0


	area_mask = (areas > upperThresh)
	label_img[area_mask[label_img]] = 0

	# print np.ndarray.dtype(label_img)
	label_img[np.where(label_img > 0)] = 2**16

	return label_img

# def noiseVis(threshImg, ks):
# 	se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# 	se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
# 	mask = cv2.morphologyEx(threshImg, cv2.MORPH_CLOSE, se2)
# 	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
#
# 	return mask

def noiseVis(threshImg, ks):
	kernelImg = cv2.morphologyEx(threshImg, cv2.MORPH_OPEN, np.ones((ks,ks)))
	ret,kernelImg = cv2.threshold(kernelImg, 0, 255, cv2.THRESH_BINARY)
	# kernelImg = cv2.erode(kernelImg, (ks,ks), iterations = 6)

	return kernelImg

def main():
	cfgfile = open("saar.ini",'r')
	config = ConfigParser.ConfigParser()
	config.read('saar.ini')

	try:
		threshVal = int(config.get('Options', 'Threshold Value'))
	except:
		print "threshVal not found in config file, did you run getParameters.py?"
	try:
		p = int(config.get('Options', 'Remove Noise Kernel Size'))
	except:
		print "kernel value not found in config file, did you run getThreshold.py?"
	try:
		lowerSizeVal = int(config.get('Options', 'Filter Size Lower Bound'))
		upperSizeVal = int(config.get('Options', 'Filter Size Upper Bound'))
	except:
		print "size values not found in config file, did you run getParameters.py?"


	startMain = timer()
	inputPath = sys.argv[1]
	outDir = sys.argv[2]

	images = sorted(glob.glob(inputPath + '*.tif*'))
	print len(images)
	for each in range(len(images)):


		imagePath = images[int(each)]
		img = cv2.imread(imagePath, -1)
		img = np.uint8(img)

		outImg = adjustThresh(img, threshVal)

		outImg = noiseVis(outImg,p)

		outImg = adjustSizeFilter(outImg, lowerSizeVal, upperSizeVal)

		tifffile.imsave(outDir + str(os.path.basename(imagePath)), outImg)
		print each

	endClean = timer() - startMain
	print "time, mass, single: " + str(endClean)
if __name__ == "__main__":
	main()
