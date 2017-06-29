import glob
import cv2
import numpy as np
import tifffile
import os
import sys

folder = sys.argv[1]

paths = glob.glob(folder + '*.tif')

stack = [tifffile.imread(path) for path in paths]

for i, img in enumerate(stack):
	img = np.flipud(img)
	tifffile.imsave('flipped_emMended_final3/' + str(os.path.basename(paths[i])), img)
