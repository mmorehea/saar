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

from skimage import measure
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

def adjustThresh(originalImg, value):
	ret,thresh1 = cv2.threshold(originalImg, int(value), 255, cv2.THRESH_BINARY)
	kernel = np.ones((3,3),np.uint8)

	thresh1 = cv2.dilate(thresh1, kernel, 1)

	thresh1 = np.uint8(ndimage.morphology.binary_fill_holes(thresh1))
	ret,thresh1 = cv2.threshold(thresh1, 0, 255, cv2.THRESH_BINARY)

	thresh1 = cv2.erode(thresh1, kernel, 1)

	return thresh1

def noiseVis(threshImg, ks):
	kernelImg = cv2.morphologyEx(threshImg, cv2.MORPH_OPEN, np.ones((ks,ks)))
	ret,kernelImg = cv2.threshold(kernelImg, 0, 255, cv2.THRESH_BINARY)
	kernelImg = cv2.erode(kernelImg, (ks,ks), iterations = 6)

	return kernelImg

def main():
	cfgfile = open("saar.ini",'r')
	config = ConfigParser.ConfigParser()
	config.read('saar.ini')

	try:
		threshVal = int(config.get('Options', 'threshold value'))
	except:
		print "threshVal not found in config file, did you run getThreshold.py?"
	try:
		p = int(config.get('Options', 'remove noise kernel size'))
	except:
		print "threshVal not found in config file, did you run getThreshold.py?"


	startMain = timer()
	inputPath = sys.argv[1]
	outDir = sys.argv[2]

	images = sorted(glob.glob(inputPath + '*'))
	for each in range(len(images)):
		print each
		imagePath = images[int(each)]
		img = cv2.imread(imagePath, -1)

		threshAndFill = adjustThresh(img, threshVal)

		noiseRemove = noiseVis(threshAndFill, p)

		tifffile.imsave(outDir + str(os.path.basename(imagePath)), noiseRemove)

	endClean = timer() - startMain
	print "time, mass, single: " + str(endClean)
if __name__ == "__main__":
	main()
