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

# def splitLabels(label, problemAreas):
#
def generateNewLabel(existingLabels):
	return

def refineLabels(array, emImages):
	uniqueLabels = np.unique(array)

	# This part can be multithreaded
	# -------------------------------------------------------------------------
	areasToReLabel = []
	for label in uniqueLabels:
		arr = array.copy()
		xCoords, yCoords, zCoords = np.where(arr==label)

		problemAreas = {}
		for z in np.unique(zCoords):
			img = arr[:,:,z]
			if cv2.__version__[0] == '3':
				contourImage, contours, hierarchy = cv2.findContours(np.uint8(img), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
			else:
				contours, hierarchy = cv2.findContours(np.uint8(img), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )

			centers = []
			for contour in contours:
				M = cv2.moments(contour)
				try:
					cx = int(M['m10']/M['m00'])
					cy = int(M['m01']/M['m00'])
				except:
					continue
				center = (cx, cy)
				centers.append(center)

			xForThisSlice = xCoords[np.where(zCoords==z)]
			yForThisSlice = xCoords[np.where(zCoords==z)]
			labelPointsForThisSlice = zip(xForThisSlice, yForThisSlice)

			contoursForThisSlice = []
			for i,center in enumerate(centers):
				if center in labelPointsForThisSlice:
					contoursForThisSlice.append(contours[i])

			if len(contoursForThisSlice) > 1:
				problemAreas[z] = contoursForThisSlice

		problemLabels.append((xCoords, yCoords, zCoords, problemAreas))
	# -------------------------------------------------------------------------
	blankArray = np.zeros(array.shape)
	for xCoords, yCoords, zCoords, problemAreas in problemLabels:
		fewLabels = []
		fewLabels, blankArray[np.array([xCoords, yCoords, zCoords])] = generateNewLabel(fewLabels)
		problemZs = problemAreas.keys()
		for z in problemZs:
			contoursForThisSlice = problemAreas[z]
			for contour in contoursForThisSlice:
				mask = np.zeros(emImages[0].shape,np.uint8)
				cv2.drawContours(mask,[cnt],0,255,-1)
				cpoints = np.transpose(np.nonzero(mask))
				cpoints = cv2.findNonZero(mask)

				cInd = zip(*cpoints)
				fewLabels, blankArray[cInd[0], cind[1], z] = generateNewLabel(fewLabels)

		newLabels = pathHunter(blankArray)
		
		# topImg = arr[:,:,z]
		# bottomImg = arr[:,:,z+1]
		# for contour in cotours


	for each in areasToReLabel:
		array[each] = generateNewLabel(uniqueLabels)

	return array




def main():
	inputPath = sys.argv[1]
	outDir = sys.argv[2]

	startMain = timer()
	threshPaths = sorted(glob.glob('cleaned/*.tif*'))

	threshPaths = sorted(glob.glob(inputPath + '*'))
	emPaths = sorted(glob.glob('emMended/*'))
	# testPaths = sorted(glob.glob('outfinal3/*'))

	threshImages = [cv2.imread(threshPaths[z], -1) for z in xrange(len(threshPaths))]
	emImages = [cv2.imread(emPaths[z], -1) for z in xrange(len(emPaths))]
	# testImages = [tifffile.imread(testPaths[z]).astype(np.uint8) for z in xrange(len(testPaths))]


	print("loaded")
	threshImages = np.dstack(threshImages)
	emImages = np.dstack(emImages)
	print("stacked")

	labeledImages, number = nd.measurements.label(threshImages)
	print number
	print("labels")

	labeledImages = np.uint32(labeledImages)

	# for each in range(len(emImages)):
	# 	blankImg = np.zeros(emImages[0].shape).astype(np.uint8)
	#
	# 	testImg = testImages[each]
	# 	emImg = emImages[each]
	# 	coords = np.where(testImg==25)
	# 	blankImg[coords] = emImg[coords]
	# 	tifffile.imsave('problemEM2/' + str(os.path.basename(testPaths[each])), blankImg)
	#
	# labeledImages = refineLabels(labeledImages, emImages)
	# print("refined")


	for each in range(emImages.shape[2]):
		print each
		img = labeledImages[:,:,each].astype(np.uint8)

		# for label in np.unique(img):

		tifffile.imsave(outDir + str(os.path.basename(threshPaths[each])), img)






if __name__ == "__main__":
	main()
