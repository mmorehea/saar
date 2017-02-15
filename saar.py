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

	blank = np.zeros(img.shape)
	kernel = np.ones((3,3),np.uint8)
	contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
	cv2.drawContours(blank, contours, -1, (255,255,255), -1)
	blank = cv2.morphologyEx(blank, cv2.MORPH_CLOSE, kernel)
	kernel = np.ones((2,2),np.uint8)
	blank = cv2.erode(blank, kernel, 1)
	cv2.imshow("window title", blank)
	cv2.waitKey()










cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
