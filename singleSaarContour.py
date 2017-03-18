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

import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import functools

NUMBERCORES = multiprocessing.cpu_count()
print "Found " + str(NUMBERCORES) + " number of cores. Using " + str(NUMBERCORES - 1) + "."
NUMBERCORES -= 1

def adjustThresh(originalImg, value):
	ret,thresh1 = cv2.threshold(originalImg, int(value), 255, cv2.THRESH_BINARY)
	return thresh1

def processEntireStack(path, emPaths):
	pool = ThreadPool(NUMBERCORES)
	emImages = [cv2.imread(emPaths[z], -1) for z in xrange(len(emPaths))]
	result = pool.map(contourAndErode, emImages)
	return result

def contourAndErode(threshImg):
	blank = np.zeros(threshImg.shape)
	kernel = np.ones((3,3),np.uint8)
	if cv2.__version__[0] == '3':
		contourImage, contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
	else:
		contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
	cv2.drawContours(blank, contours, -1, (255,255,255), -1)
	kernel = np.ones((10,10),np.uint8)
	blank = cv2.morphologyEx(blank, cv2.MORPH_CLOSE, kernel)

	blank = cv2.erode(blank, kernel, 1)
	return blank

def main():
	startMain = timer()
	emFolderPath = sys.argv[1]
	emPaths = sorted(glob.glob(emFolderPath +'*'))

		
	blank = processEntireStack('ok', emPaths)
	blank = np.uint8(np.dstack(blank))
	
	
	for each in xrange(blank.shape[2]):
		print each
		img = blank[:,:,each]
		tifffile.imsave('contour/' + str(each).zfill(4) + '.tif', img)

	
	endClean = timer() - startMain
	print "time, contour, single: " + str(endClean)
if __name__ == "__main__":
	main()
