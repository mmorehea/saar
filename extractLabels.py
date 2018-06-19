import cv2
import numpy as np
import os
import sys
import glob
import code
import tifffile
from timeit import default_timer as timer
#import ConfigParser

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
	
	emImages = [tifffile.imread(threshPaths[z]) for z in range(len(threshPaths))]
	code.interact(local=locals())
	images = np.dstack(emImages)
	blank = np.zeros(images.shape, dtype=np.uint32)
	labelList = [150391,202863,393889,1397,2334333,93924,261413]	
	for each in labelList:
		indices = np.where(images == each)
		blank[indices] = each
		
	
	
	for each in range(blank.shape[2]):
		print(each)
		img = blank[:,:,each]
		code.interact(local=locals())
		tifffile.imsave('extractedLabels/' + str(each).zfill(4) + '.tif', img)
	endClean = timer() - startMain
	print("time, labels, single: " + str(endClean))



if __name__ == "__main__":
	main()
