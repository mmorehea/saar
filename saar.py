import cv2
import numpy as np
import os
import sys
import glob
import code

def adjustThresh(originalImg, value):
	ret,thresh1 = cv2.threshold(originalImg, int(value), 255, cv2.THRESH_BINARY)
	return thresh1

def nothing(x):
    pass

def main():
	#emFolderPath = "cropedEM/"
	#emPaths = sorted(glob.glob(emFolderPath +'*'))

	#emImages = [cv2.imread(emPaths[z], -1) for z in xrange(len(emPaths))]
	#emImages = np.dstack(emImages)
	em = "cropedEM/Crop_mendedEM-0000.tiff"
	img = cv2.imread(em, -1)
	oldThresh = 200
	cv2.namedWindow('image')
	#code.interact(local=locals())

	# create trackbars for picking threshold
	cv2.createTrackbar('Threshold', 'image', 0, 255, nothing)
	threshImg = img
	while(1):
		cv2.imshow('image', threshImg)
		k = cv2.waitKey(1)
		if k == 32:
			break


			# get current positions of four trackbars
		r = cv2.getTrackbarPos('Threshold','image')
		if (r != oldThresh):
			oldThresh = r
			threshImg = adjustThresh(img, r)

	






cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
