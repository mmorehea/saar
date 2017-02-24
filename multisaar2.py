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

import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import functools

NUMBERCORES = multiprocessing.cpu_count()
print "Found " + str(NUMBERCORES) + " number of cores. Using " + str(NUMBERCORES - 1) + "."
NUMBERCORES -= 1

def findBBDimensions(listofpixels):
	if len(listofpixels) == 0:
		return None
	else:
		xs = [x[0] for x in listofpixels]
		ys = [y[1] for y in listofpixels]

		minxs = min(xs)
		maxxs = max(xs)

		minys = min(ys)
		maxys = max(ys)

		dx = max(xs) - min(xs)
		dy = max(ys) - min(ys)

		return [minxs, maxxs+1, minys, maxys+1], [dx, dy]

def adjustThresh(originalImg, value):
	ret,thresh1 = cv2.threshold(originalImg, int(value), 255, cv2.THRESH_BINARY)
	kernel = np.ones((2,2),np.uint8)
	#thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
	thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
	return thresh1

def nothing(x):
    pass

def processEntireStack(path, threshValue, emPaths):
	pool = ThreadPool(NUMBERCORES)
	emImages = [cv2.imread(emPaths[z], -1) for z in xrange(100)] # len(emPaths))]
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

def transformBlob(blob, displacement):
	dx, dy = displacement

	transformedBlob = [(point[0] + dx, point[1] + dy) for point in blob]

	return transformedBlob

def cleanLabels(img):
	kernel = np.ones((14,14),np.uint8)
	blankResult = np.zeros(img.shape, dtype=np.uint16)
	uniqueLabels = np.unique(img)[1:]
	for lab in uniqueLabels:
		indices = np.where(img==lab)
		if (indices[0].size < 10):
			#print "small label detected, skipping..."
			continue
		blob = zip(indices[0], indices[1])
		box, dimensions = findBBDimensions(blob)

		offset = [-10, 10, -10, 10]
		if (box[0] + offset[0]) < 0:
			offset[0] = 0 - box[0]
		if (box[1] + offset[1]) > img.shape[0]:
			offset[1] = img.shape[0] - box[1]
		if (box[2] + offset[2]) < 0:
			offset[2] = 0 - box[2]
		if (box[3] + offset[3]) > img.shape[1]:
			offset[3] = img.shape[1] - box[3]
		window = img[box[0] + offset[0]:box[1] + offset[1],box[2] + offset[2]:box[3] + offset[3]]
		localIndices = np.where(window==lab)
		blankImg = np.zeros(window.shape)

		blankImg[localIndices] = 99999

		blankImg = cv2.dilate(blankImg, kernel, 2)
		blankImg = cv2.erode(blankImg, kernel, 1)
		nonz = np.nonzero(blankImg)
		localBlob = zip(nonz[0],nonz[1])


		finalBlob = transformBlob(localBlob, (box[0] + offset[0], box[2] + offset[2]))

		blankResult[zip(*finalBlob)] = lab

	return blankResult

def main():
	emFolderPath = sys.argv[1]
	emPaths = sorted(glob.glob(emFolderPath +'*'))
	outputFolderPath = sys.argv[2]

	em = emPaths[0]
	img = cv2.imread(em, -1)
	oldThresh = 200
	cv2.namedWindow('image')

	# create trackbars for picking threshold
	cv2.createTrackbar('Threshold', 'image', 0, 255, nothing)
	threshImg = img
	while(1):
		try:
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
	startMain = timer()
	print "Contouring entire stack..."
	blank = processEntireStack('ok', oldThresh, emPaths)
	blank = np.dstack(blank)
	endContourStack = timer() - startMain
	start = timer()
	print "Finding connection..."
	labels = nd.measurements.label(blank)
	labels = labels[0]
	endConnections = timer() - start

	print "Cleaning labels..."
	pool = ThreadPool(NUMBERCORES)
	cleanLabels(np.dsplit(labels, labels.shape[2])[0])

	labels = np.dstack(pool.map(cleanLabels, np.dsplit(labels, labels.shape[2])))
	labels = np.uint16(labels)
	endClean = timer() - start

	print "Writing file..."
	start = timer()
	for each in xrange(labels.shape[2]):
		print each
		#code.interact(local=locals())
		img = labels[:,:,each]
		tifffile.imsave(outputFolderPath + str(each).zfill(4) + '.tif', img)
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
