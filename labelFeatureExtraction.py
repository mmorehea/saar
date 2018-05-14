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

rows = []
startPoint = 0
count = 1

if os.path.exists('featureSave.p'):
	with open('featureSave.p', 'rb') as pickleFile:
   		startPoint, count = pickle.load(pickleFile)


start = timer()
for i, prop in enumerate(properties):
	if i < startPoint:
		continue

	# Skipping the 50th label for now because it's particularly huge and takes forever to process
	if i == 49:
		continue

	print(str(i+1) + '/' + str(len(properties)))
	if prop.label in itemlist:
		print('\t' + str(count) + '/' + str(len(itemlist)))

		newRow = [prop.label, prop.area, prop.bbox, prop.bbox_area, prop.centroid, prop.equivalent_diameter, prop.extent, prop.filled_area, prop.max_intensity, prop.mean_intensity, prop.min_intensity]
		
		# place any tuple elements in separate columns
		for x, item in enumerate(newRow):
			if type(item) is tuple:
				del newRow[x]
				newRow[x:x] = [item[q] for q in range(len(item))]
		
		if len(rows) == 0:
			rows = np.array(newRow)
		else:
			rows = np.vstack((rows, newRow))


		count += 1
	if (i+1) % 300 == 0:
		del properties
		properties = regionprops(labelStack, emStack)
	if (i+1) % 100 == 0:
		with open('labelFeatures.csv', 'a') as ff:
			writer = csv.writer(ff)
			writer.writerows(rows)
		startPoint = i + 1
		rows = []
		print('saving...')
		pickle.dump((startPoint, count), open('featureSave.p', 'wb'))
	elif i + 1 == len(properties):
		with open('labelFeatures.csv', 'a') as ff:
			writer = csv.writer(ff)
			writer.writerows(rows)


end = timer()
print(end-start)



