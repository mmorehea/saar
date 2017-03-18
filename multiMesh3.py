from marching_cubes import march
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
import sys

SCALEX = 10.0
SCALEY = 10.0
SCALEZ = 1.0

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

	return [minxs, maxxs+1, minys, maxys+1, minzs, maxzs+1], [dx, dy, dz]

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
			f.write("v %.2f %.2f %.2f \n" % ((box[0] * SCALEX) + (v[2] * SCALEX), (box[2] * SCALEY) + (v[1] * SCALEY), (box[3] * SCALEZ) + v[0] * 5.454545))
		for n in normals:
			f.write("vn %.2f %.2f %.2f \n" % (n[2], n[1], n[0]))
		for face in faces:
			f.write("f %d %d %d \n" % (face[0]+1, face[1]+1, face[2]+1))


def main():
	q = queue.Queue()
	meshes = sys.argv[2]
	alreadyDone = glob.glob(meshes + "*")

	labelsFolderPath = sys.argv[1]
	labelsPaths = sorted(glob.glob(labelsFolderPath +'*'))
	#code.interact(local=locals())
	labelStack = [tifffile.imread(labelsPaths[z]) for z in range(len(labelsPaths))]
	labelStack = np.dstack(labelStack)
	print("Loaded data...")
	labels = np.unique(labelStack)[1:]
	print("Found labels...")
	print("firstlabel: " + str(labels[0]))
	print("Number of labels", str(len(labels)))

	for each in labels:
		calcMesh(each, labelStack, meshes)


if __name__ == "__main__":
	main()
