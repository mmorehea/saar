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
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



rows = []
with open('labelFeatures.csv') as ff:
	reader = csv.reader(ff, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
	for row in reader:
		rows.append([float(x) for x in row])

classifiedLabels = []
with open('minedMeshLabel.csv') as ff:
	reader = csv.reader(ff, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
	for label in reader:
		classifiedLabels.append(label)

minedLabels = [float(z[0].split('::')[-1][:-4]) for z in classifiedLabels]

labelClasses = [int(each[1]) for each in classifiedLabels]
classDict = {}
for i, each in enumerate(minedLabels):
	# filter out merged category
	if labelClasses[i] != 3:
		classDict[int(each)] = labelClasses[i]

data = [row for row in rows if int(row[0]) in classDict.keys()]

unknownData = [row for row in rows if int(row[0]) not in classDict.keys()]

target = np.array([classDict[int(each[0])] for each in data])

# Eliminate the first column since it is the label ID
data = [row[1:] for row in data]
unknownData = [row[1:] for row in unknownData]

# Eliminate bbox coordinates
data = [row[:1] + row[5:] for row in data]
unknownData = [row[:1] + row[5:] for row in unknownData]

# Eliminate centroid
data = [row[:2] + row[4:] for row in data]
unknownData = [row[:2] + row[4:] for row in unknownData]

data = np.array(data)
unknownData = np.array(unknownData)

# Preprocessing and classification pipeline
lda = LinearDiscriminantAnalysis().fit(data, target)
data = lda.transform(data)

clf = svm.SVC()
clf.fit(data,target)

# scores = cross_val_score(clf, data, target, cv=5)
# print(scores)

unknownData = lda.transform(unknownData)
predictions = clf.predict(unknownData)

code.interact(local=locals())

# Notes
# Eliminating centroid and bbox coordinates had NO effect on accuracy
# Linear discriminant analysis seems to improve accuracy

