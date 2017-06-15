import sys
import code
import tifffile
import numpy as np
import glob
from numpy import load
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
from timeit import default_timer as timer
import threading
import os
import pickle
import math

def findBBDimensions(listOfPixels):
	xs = listOfPixels[0]
	ys = listOfPixels[1]
	zs = listOfPixels[2]

	minxs = min(xs)
	maxxs = max(xs)

	minys = min(ys)
	maxys = max(ys)

	minzs = min(zs)
	maxzs = max(zs)

	dx = maxxs - minxs
	dy = maxys - minys
	dz = maxzs - minzs

	return [minxs-2, maxxs+2, minys-2, maxys+2, minzs-2, maxzs+2], [dx, dy, dz]

def trackSize(labelStack, axis, start):
	tracker = {}
	for i in xrange(labelStack.shape[axis]):
		end = timer()
		print(str(i) + "/" + str(labelStack.shape[axis]) + " time: " + str(end-start))

		if axis == 0:
			img = labelStack[i,:,:]
		elif axis == 1:
			img = labelStack[:,i,:]
		elif axis == 2:
			img = labelStack[:,:,i]

		idList = np.unique(img)

		for each in tracker.keys():
			tracker[each][2] += 1
			if tracker[each][2] > 25:
				if tracker[each][1] - tracker[each][0] < 500:
					del tracker[each]

		for itm in idList:

			if itm not in tracker.keys():
				tracker[itm] = [i, 0, 0]
			else:
				if i > tracker[itm][1]:
					tracker[itm][1] = i + 1
					tracker[itm][2] = 0

	finalList = [t for t in tracker.keys() if tracker[t][1] - tracker[t][0] > 500]
	return finalList

def main():
	start = timer()
	labelsFolderPath = sys.argv[1]
	code.interact(local=locals())
	labelsPaths = sorted(glob.glob(labelsFolderPath +'*'))
	labelStack = [tifffile.imread(labelsPaths[z]) for z in range(len(labelsPaths))]
	code.interact(local=locals())
	labelStack = np.dstack(labelStack)
	end = timer()
 	print("Loaded data... time: " + str(end-start))
	code.interact(local=locals())
	print("X Direction...")
	finalListX = trackSize(labelStack, 0, start)
	print("Y Direction...")
	finalListY = trackSize(labelStack, 1, start)
	print("Z Direction...")
	finalListZ = trackSize(labelStack, 2, start)

	finalList = list(set(finalListX) | set(finalListY) | set(finalListZ))
	print(end-start)

	code.interact(local=locals())
	pickle.dump(finalList, open('outfile.p', 'wb'))

if __name__ == "__main__":
	main()

	# idList = np.unique(labelStack)
	# print("Loaded ID list...")
	#
	# goodLabels = []
	# for i, itm in enumerate(idList):
	# 	print(str(i) + "/" + str(len(idList)))
	# 	blob = np.where(labelStack==itm)
	#
	# 	box, dimensions = findBBDimensions(blob)
	#
	# 	distance = math.sqrt(dimensions[0]**2 + dimensions[1]**2 + dimensions[2]**2)
	#
	# 	if distance > 500:
	# 		goodLabels.append(itm)
	#
	# code.interact(local=locals())
	# pickle.dump(goodLabels, open('outfile.p', 'wb'))

	# print("X Direction...")
	# tracker = {}
	# for i in xrange(labelStack.shape[0]):
	# 	end = timer()
	# 	print(str(i) + "/" + str(labelStack.shape[0]) + " time: " + str(end-start))
	# 	imgx = labelStack[i,:,:]
	# 	idList = np.unique(imgx)
	#
	# 	for each in tracker.keys():
	# 		tracker[each][2] += 1
	# 		if tracker[each][2] > 25:
	# 			if tracker[each][1] - tracker[each][0] < 500:
	# 				del tracker[each]
	#
	# 	for itm in idList:
	#
	# 		if itm not in tracker.keys():
	# 			tracker[itm] = [i, 0, 0]
	# 		else:
	# 			if i > tracker[itm][1]:
	# 				tracker[itm][1] = i + 1
	# 				tracker[itm][2] = 0
	# finalListX = set([tracker[t] for t in tracker.keys() if tracker[t][1] - tracker[t][0] > 500])
	#
	# print("Y Direction...")
	# tracker = {}
	# for i in xrange(labelStack.shape[1]):
	# 	end = timer()
	# 	print(str(i) + "/" + str(labelStack.shape[1]) + " time: " + str(end-start))
	# 	imgy = labelStack[:,i,:]
	# 	idList = np.unique(imgy)
	#
	# 	for each in tracker.keys():
	# 		tracker[each][2] += 1
	# 		if tracker[each][2] > 25:
	# 			if tracker[each][1] - tracker[each][0] < 500:
	# 				del tracker[each]
	#
	# 	for itm in idList:
	#
	# 		if itm not in tracker.keys():
	# 			tracker[itm] = [i, 0, 0]
	# 		else:
	# 			if i > tracker[itm][1]:
	# 				tracker[itm][1] = i + 1
	# 				tracker[itm][2] = 0
	# finalListY = set([tracker[t] for t in tracker.keys() if tracker[t][1] - tracker[t][0] > 500])
	#
	# print("Z Direction...")
	# tracker = {}
	# for i in xrange(labelStack.shape[2]):
	# 	end = timer()
	# 	print(str(i) + "/" + str(labelStack.shape[2]) + " time: " + str(end-start))
	# 	imgz = labelStack[:,:,i]
	# 	idList = np.unique(imgz)
	#
	# 	for each in tracker.keys():
	# 		tracker[each][2] += 1
	# 		if tracker[each][2] > 25:
	# 			if tracker[each][1] - tracker[each][0] < 500:
	# 				del tracker[each]
	#
	# 	for itm in idList:
	#
	# 		if itm not in tracker.keys():
	# 			tracker[itm] = [i, 0, 0]
	# 		else:
	# 			if i > tracker[itm][1]:
	# 				tracker[itm][1] = i + 1
	# 				tracker[itm][2] = 0
	# finalListZ = set([tracker[t] for t in tracker.keys() if tracker[t][1] - tracker[t][0] > 500])
