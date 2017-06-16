from __future__ import division
import sys
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
from timeit import default_timer as timer

# NUMBERCORES = multiprocessing.cpu_count()
# print("Found " + str(NUMBERCORES) + " number of cores. Using 2.")
# NUMBERCORES = 2

SCALEX = 10.0
SCALEY = 10.0
SCALEZ = 1.0
labelStack = []
meshes = []

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

def calcMesh(label):

	indices = np.where(labelStack==label)
	box, dimensions = findBBDimensions(indices)


	window = labelStack[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
	localIndices = np.where(window==label)
	blankImg = np.zeros(window.shape, dtype=bool)
	blankImg[localIndices] = 1

	vertices, normals, faces = march(blankImg.transpose(), 1)  # zero smoothing rounds

	with open(meshes + str(label)+".obj", 'w') as f:
		f.write("# OBJ file\n")
		for v in vertices:
			f.write("v %.2f %.2f %.2f \n" % ((box[0] * SCALEX) + (v[2] * SCALEX), (box[2] * SCALEY) + (v[1] * SCALEY), (box[4] * SCALEZ) + v[0] * 5.454545))
		for n in normals:
			f.write("vn %.2f %.2f %.2f \n" % (n[2], n[1], n[0]))
		for face in faces:
			f.write("f %d %d %d \n" % (face[0]+1, face[1]+1, face[2]+1))


def main():
	q = queue.Queue()
	start = timer()
	global meshes
	meshes = sys.argv[2]

	alreadyDone = glob.glob(meshes + "*.obj")

	alreadyDone = [i.split("\\") for i in alreadyDone]
	print(alreadyDone)

	labelsFolderPath = sys.argv[1]
	labelsPaths = sorted(glob.glob(labelsFolderPath +'*'))
	#code.interact(local=locals())
	global labelStack
	labelStack = [tifffile.imread(labelsPaths[z]) for z in range(len(labelsPaths))]
	labelStack = np.dstack(labelStack)

	print("Loaded data...")
	with open ('outfile.npy', 'rb') as fp:
		itemlist = list(np.load(fp))
		itemlist = itemlist[1:]

	# itemlist = np.unique(labelStack)[1:]

	print("Found labels...")
	print("firstlabel: " + str(itemlist[0]))
	print("Number of labels", str(len(itemlist)))

	# pool = ThreadPool(NUMBERCORES)

	itemlist = [itm for itm in itemlist if itm not in alreadyDone]
	
	# for i, _ in enumerate(pool.imap_unordered(calcMesh, itemlist), 1):
	# 	sys.stderr.write('\rdone {0:%}'.format(i/len(itemlist)))
	
	for i, itm in enumerate(itemlist):
		calcMesh(itm)
		end = timer()
		print(str(i+1) + "/" + str(len(itemlist)) + " time: " + str(end-start))

if __name__ == "__main__":
	main()
