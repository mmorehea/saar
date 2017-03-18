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


def main():
	startMain = timer()
	threshPath = sys.argv[1]
	threshPaths = sorted(glob.glob(threshPath +'*'))
	
	emImages = [cv2.imread(threshPaths[z], -1) for z in xrange(len(threshPaths))]
	images = np.dstack(emImages)
	blank = np.zeros(images.shape)
	labelList = [2690, 24278]	
	for each in labelList:
		indices = np.where(images == each)
		blank[indices] = each
		
	blank = np.uint16(blank)
	
	for each in xrange(blank.shape[2]):
		print each
		img = blank[:,:,each]
		tifffile.imsave('extractedLabels/' + str(each).zfill(4) + '.tif', img)
	endClean = timer() - startMain
	print "time, labels, single: " + str(endClean)



if __name__ == "__main__":
	main()
