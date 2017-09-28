import glob
import numpy as np
import tifffile
import os
import sys

folder = sys.argv[1]

outDir = sys.argv[2]

paths = glob.glob(folder + '*.tif')

stack = [tifffile.imread(path) for path in paths]

for i, img in enumerate(stack):
	img = np.flipud(img)
	tifffile.imsave(outDir + str(os.path.basename(paths[i])), img)
