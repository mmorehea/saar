import sys
import code
import tifffile
import numpy as np
import glob
from numpy import load
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import queue
import threading
import os
import pickle

def main():
	labelsFolderPath = sys.argv[1]
	labelsPaths = sorted(glob.glob(labelsFolderPath +'*'))
	labelStack = [tifffile.imread(labelsPaths[z]) for z in range(len(labelsPaths))]
	# labelStack = np.dstack(labelStack)
	print("Loaded data...")
	tracker = {}

	for i, img in enumerate(labelStack):
		print i
		idList = np.unique(img)

		for itm in idList:
			if itm not in tracker.keys():
				tracker[itm] = [i, 0]
			else:
				if i > tracker[itm][1]:
					tracker[itm][1] = i + 1

	finalList = [tracker[t] for t in tracker.keys() if tracker[t][1] - tracker[t][0] > 500]
	code.interact(local=locals())
	pickle.dump(finalList, open('outfile.p', 'wb'))


if __name__ == "__main__":
	main()
