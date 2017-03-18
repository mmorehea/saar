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


def cleanLabels(img):
	kernel = np.ones((3,3),np.uint8)
	blankResult = np.zeros(img.shape, dtype=np.uint32)
	uniqueLabels = np.unique(img)[1:]
	sizeLabels = len(uniqueLabels)
	for ii,lab in enumerate(uniqueLabels):
		#print str(ii) + " / " + str(sizeLabels)
		indices = np.where(img==lab)

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

		blankResult[zip(*finalBlob)] = ii

	return blankResult

def transformBlob(blob, displacement):
	dx, dy = displacement

	transformedBlob = [(point[0] + dx, point[1] + dy) for point in blob]

	return transformedBlob

def main():
	startMain = timer()
	imageNum = sys.argv[2]
	imagePath = 'saarData/' # labels/
	images = sorted(glob.glob(imagePath + '*.npy'))
	imagePath = images[int(imageNum)]
	img = np.load(imagePath)

	cleanImage = cleanLabels(img)

	tifffile.imsave('clean/' + str(os.path.basename(imagePath)[:-4]) + '.tif', cleanImage)

	endClean = timer() - startMain
	print "time, clean, single: " + str(endClean)


if __name__ == "__main__":
	main()
