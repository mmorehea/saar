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
import csv
from timeit import default_timer as timer
from sklearn import svm


rows = []
with open('labelFeatures.csv') as ff:
	reader = csv.reader(ff, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
	for row in reader:
		rows.append([float(x) for x in row])	

code.interact(local=locals())

X = rows[:int(len(rows))/2]
y = rows[int(len(rows))/2:]

clf = svm.SVC()

clf.fit(X,y)


