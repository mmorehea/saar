import numpy as np
import os
import sys
import glob
import code
import tifffile
from timeit import default_timer as timer
import ConfigParser



def main():
	startMain = timer()
	threshPath = sys.argv[1]
	threshPaths = sorted(glob.glob(threshPath +'*'))
	
	emImages = [tifffile.imread(threshPaths[z]) for z in xrange(len(threshPaths))]
	images = np.dstack(emImages)
	blank = np.zeros(images.shape)
	labelList = [34221,48784,41281,89215,62078,88538,74564,29411,66598,90149,37062,41570,46280,79622,56732,66891,91509,80561,61940,82571,91968,85449,90470,69924]
	ids = np.unique(images)
	for each in ids:
		if each not in labelList:
			indices = np.where(images == each)
			blank[indices] = each
		
	blank = np.uint32(blank)
	
	for each in xrange(blank.shape[2]):
		print each
		img = blank[:,:,each]
		tifffile.imsave('extractedLabels/' + str(each).zfill(4) + '.tif', img)
	endClean = timer() - startMain
	print "time, labels, single: " + str(endClean)



if __name__ == "__main__":
	main()
