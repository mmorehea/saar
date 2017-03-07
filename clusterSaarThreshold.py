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

def adjustThresh(originalImg, value):
	ret,thresh1 = cv2.threshold(originalImg, int(value), 255, cv2.THRESH_BINARY)
	#kernel = np.ones((4,4),np.uint8)
	#thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
	#thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
	return thresh1

def main():
	startMain = timer()
	imageNum = sys.argv[2]
	imagePath = 'mendedEM/'
	images = glob.glob(imagePath + '*')
	cfgfile = open("saar.ini",'r')
	config = ConfigParser.ConfigParser()
	config.read('saar.ini')
	try:
		threshVal = int(config.get('Options', 'threshold value'))
	except:
		print "threshVal not found in config file, did you run getThreshold.py?"	
	imagePath = images[int(imageNum)]
	img = cv2.imread(imagePath, -1)
	threshImg = np.uint8(adjustThresh(img, threshVal))
	
	tifffile.imsave('thresh/' + str(os.path.basename(imagePath)), threshImg)
	
	endClean = timer() - startMain
	print "time, thresh, single: " + str(endClean)
if __name__ == "__main__":
	main()
