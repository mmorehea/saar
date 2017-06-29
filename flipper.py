import glob
import cv2
import numpy as np
import tifffile

paths = glob.glob('cap8bit/*.tif')

stack = [tifffile.imread(path) for path in paths]

img = stack[0]

img2 = np.flipud(img)

tifffile.imsave('a/a.tif', img)
tifffile.imsave('a/b.tif', img2)
