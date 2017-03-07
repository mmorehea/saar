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

def processEntireStack(path, threshValue, emPaths):
	pool = ThreadPool(NUMBERCORES)
	emImages = [cv2.imread(emPaths[z], -1) for z in xrange(len(emPaths))]
	result = pool.map(functools.partial(adjustThresh, value = threshValue), emImages)
	return result

def main():
	startMain = timer()
	emFolderPath = sys.argv[1]
	emPaths = sorted(glob.glob(emFolderPath +'*'))
	cfgfile = open("saar.ini",'r')
	config = ConfigParser.ConfigParser()
	config.read('saar.ini')
	try:
		threshVal = int(config.get('Options', 'threshold value'))
	except:
		print "threshVal not found in config file, did you run getThreshold.py?"	
	
		
	blank = processEntireStack('ok', threshVal, emPaths)
	blank = np.uint8(np.dstack(blank))
	
	
	for each in xrange(blank.shape[2]):
		print each
		img = blank[:,:,each]
		tifffile.imsave('thresh/' + str(each).zfill(4) + '.tif', img)
		
	endClean = timer() - startMain
	print "time, thresh, single: " + str(endClean)
if __name__ == "__main__":
	main()
