import cv2
import numpy as np
import os
import sys
import glob
import code
import tifffile
from timeit import default_timer as timer
import ConfigParser
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

from scipy import ndimage as nd

from skimage import measure
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

NUMBERCORES = multiprocessing.cpu_count()
print "Found " + str(NUMBERCORES) + " number of cores. Using " + str(NUMBERCORES - 1) + "."
NUMBERCORES -= 1


def orderContoursBySize(contours):
	sizes = []
	for contour in contours:
		area = cv2.contourArea(contour)
		sizes.append(area)
	indices = sorted(range(len(sizes)), key = lambda k: sizes[k])

	orderedSizes = []
	orderedContours = []
	for ind in indices:
		orderedSizes.append(sizes[ind])
		orderedContours.append(contours[ind])

	return orderedContours, orderedSizes

def filterSize(img):
	# if cv2.__version__[0] == '3':
	# 	contourImage, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
	# else:
	# 	contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )

	label_img, cc_num = nd.label(img)
	objs = nd.find_objects(label_img)
	areas = nd.sum(img, label_img, range(cc_num+1))

	indices = sorted(range(len(areas)), key = lambda k: areas[k])

	orderedAreas = []

	for ind in indices:
		orderedAreas.append(areas[ind])


	# contours, sizes = orderContoursBySize(contours)

	lowerThresh = orderedAreas[int((float(firstPercentile)/100) * len(orderedAreas))]
	upperThresh = orderedAreas[int((float(secondPercentile)/100) * len(orderedAreas))]

	area_mask = (areas < lowerThresh)
	label_img[area_mask[label_img]] = 0

	area_mask = (areas > upperThresh)
	label_img[area_mask[label_img]] = 0

	label_img[np.where(label_img > 0)] = 255

	# for i, obj in enumerate(orderedObjs):
	# 	if i < lowerThreshIndex or i > upperThreshIndex:
	# 		mask = np.zeros(img.shape,np.uint8)
	# 		cv2.drawContours(mask,[contour],0,255,-1)
	# 		pixelpoints = np.transpose(np.nonzero(mask))
	# 		for point in pixelpoints:
	# 			img[point[0],point[1]] = 0
	return label_img

firstPercentile = sys.argv[3] # Recommended: 50
secondPercentile = sys.argv[4] # Recommended: 99
def main():
	inputPath = sys.argv[1]
	outDir = sys.argv[2]


	startMain = timer()
	threshPaths = sorted(glob.glob(inputPath +'*.tif*'))

	emImages = [cv2.imread(threshPaths[z], -1) for z in xrange(len(threshPaths))]
	print("loaded")

	#pool = ThreadPool(NUMBERCORES)
	#emResults = pool.map(filterSize, emImages)

	for ii, img in enumerate(emImages):
		img = filterSize(img)
		tifffile.imsave(outDir + str(os.path.basename(threshPaths[ii])), img)
		print ii



if __name__ == "__main__":
	main()
