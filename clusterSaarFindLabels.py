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
	threshPaths = sorted(glob.glob('cleaned/*.tif*'))

	emImages = [cv2.imread(threshPaths[z], -1) for z in xrange(len(threshPaths))]
	print("loaded")
	emImages = np.dstack(emImages)
	print("stacked")

	emImages, number = nd.measurements.label(emImages)
	print number
	print("labels")

	emImages = np.uint32(emImages)


	for each in range(emImages.shape[2]):
		print each
		img = emImages[:,:,each]
		tifffile.imsave('cleanedLabels32/' + str(os.path.basename(threshPaths[each])), img)






if __name__ == "__main__":
	main()
