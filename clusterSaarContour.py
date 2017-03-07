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

def contourAndErode(threshImg):
	blank = np.zeros(threshImg.shape)

	if cv2.__version__[0] == '3':
		contourImage, contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
	else:
		contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
	cv2.drawContours(blank, contours, -1, (255,255,255), -1)
	kernel = np.ones((3,3),np.uint8)
	blank = cv2.morphologyEx(blank, cv2.MORPH_CLOSE, kernel)
	blank = cv2.erode(blank, kernel, 1)
	return blank

def main():
	startMain = timer()
	imageNum = sys.argv[2]
	imagePath = 'thresh/'
	images = sorted(glob.glob(imagePath + '*'))
	imagePath = images[int(imageNum)]
	img = cv2.imread(imagePath, -1)
	contourImage = np.uint8(contourAndErode(img))
	
	tifffile.imsave('contour/' + str(os.path.basename(imagePath)), contourImage)
	
	endClean = timer() - startMain
	print "time, contour, single: " + str(endClean)
if __name__ == "__main__":
	main()
