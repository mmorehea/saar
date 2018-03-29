from __future__ import division
import cv2
import numpy as np
from numpy import load
import os
import sys
import glob
import code
import tifffile
import threading
from marching_cubes import march
from timeit import default_timer as timer
import configparser
from scipy import ndimage as nd
from skimage import measure
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
from itertools import cycle
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import threading
import matplotlib.pyplot as plt
import queue
import ast
import hashlib

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
	# stack = sys.argv[1]
	# labelStack = tifffile.imread(stack)
	# print("loaded image stack")
	
	# with open ('outfile.npy', 'rb') as fp:
	# 	itemlist = np.load(fp)
	# 	itemlist = itemlist[1:]
	# itemlist = sorted(itemlist)
	# code.interact(local=locals())

	# for i, itm in enumerate(itemlist):
	# 	print(itm)
	# 	print(str(i) + '/' + str(len(itemlist)))
	
	# 	indices = np.where(labelStack==itm)

	# 	if len(indices[0]) < 10:
	# 		print('skipping')
	# 		continue
	
	meshesFolderPath = sys.argv[1]
	meshPaths = sorted(glob.glob(meshesFolderPath +'*.obj'))

	hashList = [md5(path) for path in meshPaths]
	# code.interact(local=locals())


	# meshPaths = sorted(glob.glob(meshesFolderPath +'*.tif*'))
	for i, path in enumerate(meshPaths):
		with open(path, 'r') as f:
			lines = f.readlines()
		vertices = []
		for line in lines:
			if line[0] == 'v':
				vertex = tuple([float(a) for a in line[2:-2].split(' ')])
				vertices.append(vertex)
			if line[0] == 'f':
				code.interact(local=locals())


if __name__ == "__main__":
	main()
