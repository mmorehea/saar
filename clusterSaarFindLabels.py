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
	inputPath = sys.argv[1]
	outDir = sys.argv[2]

	startMain = timer()
	# threshPaths = sorted(glob.glob('cleaned/*.tif*'))
	threshPaths = sorted(glob.glob(inputPath + '*'))

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
		img = emImages[:,:,each].astype()

		blank = np.zeros(img.shape)

		if cv2.__version__[0] == '3':
			contourImage, contours, hierarchy = cv2.findContours(np.uint8(img), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
		else:
			contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )

		centers = []
		for contour in contours:
			M = cv2.moments(contour)
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			center = (cx, cy)
			centers.append(center)

		problemLabs = []
		labs = []
		for center in centers:
			lab = img[center]
			if lab not in labs:
				labs.append(lab)
			else:
				problemLabs.append(lab)





		# for label in np.unique(img):

		tifffile.imsave(outDir + str(os.path.basename(threshPaths[each])), img)






if __name__ == "__main__":
	main()
