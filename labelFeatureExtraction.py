import code
import numpy as np
import tifffile
import glob
import subprocess
import threading
import os
import sys
import pickle
import platform
from skimage.measure import regionprops
import csv
from timeit import default_timer as timer

labels = sys.argv[1]
intensity = sys.argv[2]

labelStack = tifffile.imread(labels).astype(int)
emStack = tifffile.imread(intensity).astype(int)

with open('outfile.npy','rb') as fp:
	itemlist = np.load(fp)
	itemlist = itemlist[1:]


properties = regionprops(labelStack, emStack)


rows = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16, 17, 18])
startPoint = 0
count = 1

if os.path.exists('featureSave.p'):
	with open('featureSave.p', 'rb') as pickleFile:
   		rows, startPoint, count = pickle.load(pickleFile)


start = timer()
for i, prop in enumerate(properties):
	if i < startPoint:
		continue

	print(str(i+1) + '/' + str(len(properties)))
	if prop.label in itemlist:
		print('\t' + str(count) + '/' + str(len(itemlist)))

		newRow = [prop.label, prop.area, prop.bbox, prop.bbox_area, prop.centroid, prop.equivalent_diameter, prop.extent, prop.filled_area, prop.max_intensity, prop.mean_intensity, prop.min_intensity]
		
		# place any tuple elements in separate columns
		for x, item in enumerate(newRow):
			if type(item) is tuple:
				del newRow[x]
				newRow[x:x] = [item[i] for i in range(len(item))]
		
		rows = np.vstack((rows, newRow))


		count += 1

		if (i+1) % 100 == 0:
			startPoint = i + 1
			pickle.dump((rows, startPoint, count), open('featureSave.p', 'wb'))
		


end = timer()
print(end-start)


with open('labelFeatures.csv', 'w') as ff:
	writer = csv.writer(ff)
	writer.writerows(rows)


