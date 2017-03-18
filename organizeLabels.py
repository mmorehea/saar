import code
import tifffile
import numpy as np
import glob
from numpy import load
import cPickle as pickle
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import queue
import threading
import os
import sys

SCALEX = 5.0
SCALEY = 5.0

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

	dx = maxxs+1 - minxs
	dy = maxys+1 - minys
	dz = maxzs+1 - minzs

	return [minxs, maxxs+1, minys, maxys+1, minzs, maxzs+1], [dx, dy, dz]

def transformCoords(labelPoints, boundingBox, direction):
# for 'direction' argument, +1 is local to global, and -1 is global to local
	dx = boundingBox[0]
	dy = boundingBox[2]
	dz = boundingBox[4]

	labelPoints = (np.array([x+direction*dx for x in labelPoints[0]]), np.array([y+direction*dy for y in labelPoints[1]]), np.array([z+direction*dz for z in labelPoints[2]]))

	return labelPoints, np.array([dx, dy, dz])

def calcMesh(label, labelStack, location):

	indices = np.where(labelStack==label)
	if len(indices[0]) < 5000:
		print("Too small", str(len(indices[0])))
		return
	#print len(indices[0])

	box, dimensions = findBBDimensions(indices)


	window = labelStack[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
	localIndices = np.where(window==label)
	blankImg = np.zeros(window.shape, dtype=bool)
	blankImg[localIndices] = 1

	vertices, normals, faces = march(blankImg.transpose(), 1)  # zero smoothing rounds

	with open(location + str(label)+".obj", 'w') as f:
		f.write("# OBJ file\n")
		for v in vertices:
			f.write("v %.2f %.2f %.2f \n" % ((box[0] * SCALEX) + (v[2] * SCALEX), (box[2] * SCALEY) + (v[1] * SCALEY), v[0] * 5.454545))
		for n in normals:
			f.write("vn %.2f %.2f %.2f \n" % (n[2], n[1], n[0]))
		for face in faces:
			f.write("f %d %d %d \n" % (face[0]+1, face[1]+1, face[2]+1))


def main():
	q = queue.Queue()
	# meshes = sys.argv[2]
	# alreadyDone = glob.glob(meshes + "*")

	labelsFolderPath = sys.argv[1]
	labelsPaths = sorted(glob.glob(labelsFolderPath +'*'))

	labelStack = [tifffile.imread(labelsPaths[z]) for z in range(len(labelsPaths))]
	labelStack = np.dstack(labelStack)
	print("Loaded data...")
	labels = np.unique(labelStack)[1:]
	print("Found labels...")
	print("firstlabel: " + str(labels[0]))

	for i, label in enumerate(labels):
		labelPoints = np.where(labelStack==label)
		box, dimensions = findBBDimensions(labelPoints)

		localLabelPoints, globalCoords = transformCoords(labelPoints, box, -1)

		try:
			labelArray = np.zeros((dimensions[0], dimensions[1], dimensions[2]), dtype=bool)
			labelArray[localLabelPoints] = 1
		except: code.interact(local=locals())

		if not os.path.exists('labelData/'):
			os.mkdir('labelData')
		if not os.path.exists('labelData/' + str(i) + '/'):
			os.mkdir('labelData/' + str(i) + '/')

		np.save('labelData/' + str(i) + '/array.npy', labelArray)
		np.save('labelData/' + str(i) + '/globalCoords.npy', globalCoords)


if __name__ == "__main__":
	main()
