from __future__ import division
import sys
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
import collections

def printStack(stack, path):
	for z in xrange(stack.shape[2]):
		img = stack[:,:,z]
		tifffile.imsave(path + str(z) + '.tif', img)

def main():
	start = timer()
	labelsFolderPath = sys.argv[1]
	bigIDs = [int(a) for a in sys.argv[2].split(',')]

	labelsPaths = sorted(glob.glob(labelsFolderPath +'*.tif*'))
	labelStack = [tifffile.imread(labelsPaths[z]) for z in range(len(labelsPaths))]

	labelStack = np.dstack(labelStack)
	blankStack = np.zeros(labelStack.shape, np.uint8)
	newStack = []
	end = timer()
	print("Loaded data... time: " + str(end-start))

	for labelID in bigIDs:
		blankStack[np.where(labelStack==labelID)] = 99999

		printStack(blankStack, 'singleLabel/')

		for z in xrange(blankStack.shape[2]):
			print(z)
			img = blankStack[:,:,z]
			kernel = np.ones((3,3),np.uint8)
			img = cv2.erode(img, kernel, 2)
			newStack.append(img)


	newStack = np.dstack(newStack)
	end = timer()
	print("Created stack with only the label... time: " + str(end-start))

	labeledStack, number = nd.measurements.label(newStack)
	labeledStack = np.uint8(labeledStack)
	end = timer()
	print("Finished connected components... time: " + str(end-start))
	printStack(labeledStack, 'splitLabel/')

	# blankStackPaths = sorted(glob.glob('singleLabel/*.tif*'))
	# blankStack = [cv2.imread(each, -1) for each in blankStackPaths]
	# blankStack = np.dstack(blankStack)
	#
	# labeledStackPaths = sorted(glob.glob('splitLabel/*.tif*'))
	# labeledStack = [cv2.imread(each, -1) for each in labeledStackPaths]
	# labeledStack = np.dstack(labeledStack)

	for z in xrange(labeledStack.shape[2]):
		print(str(z) + ' time: ' + str(timer()-start))
		binaryImg = blankStack[:,:,z]
		labelImg = labeledStack[:,:,z]

		if cv2.__version__[0] == '3':
			binaryImg2, contours, hierarchy = cv2.findContours(binaryImg.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		else:
			contours, hierarchy = cv2.findContours(binaryImg.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		blobs = []
		for contour in contours:
			mask = np.zeros(labelImg.shape,np.uint8)
			cv2.drawContours(mask,[contour],0,255,-1)
			pixelpoints = np.transpose(np.nonzero(mask))
			blobs.append([(x[0],x[1]) for x in pixelpoints])

		for blob in blobs:
			window = labelImg[zip(*blob)]
			try:
				blobLabel = [a[0] for a in collections.Counter(window).most_common() if a[0] != 0][0]
			except:
				blobLabel = 0
			binaryImg[zip(*blob)] = blobLabel

		tifffile.imsave('splitLabelDilated/' + str(z) + '.tif', binaryImg)

if __name__ == "__main__":
	main()
