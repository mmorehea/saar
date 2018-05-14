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
with open('labelFeatures900.csv') as ff:
	reader = csv.reader(ff, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
	for row in reader:
		rows.append([float(x) for x in row])	

Xtrain = np.array(rows[:int(len(rows)/2)])

classLabels = ['Good', 'Bad', 'Merge', 'Split']
y = np.random.choice(classLabels, Xtrain.shape[0])

clf = svm.SVC()

clf.fit(Xtrain,y)

Xtest = np.array(rows[int(len(rows)/2):])

result = clf.predict(Xtest)

code.interact(local=locals())


