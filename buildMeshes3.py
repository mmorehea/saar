from marching_cubes import march
import code
import tifffile
import numpy as np
import glob
from numpy import load
import sys

def writeObj(filepath, vertices, normals, faces):
	with open(filepath, 'w') as f:
		f.write("# OBJ file\n")
		for v in vertices:
			f.write("v %.2f %.2f %.2f \n" % (v[0], v[1], v[2]))
		for n in normals:
			f.write("vn %.2f %.2f %.2f \n" % (n[0], n[1], n[2]))
		for face in faces:
			f.write("f %d %d %d \n" % (face[0]+1, face[1]+1, face[2]+1))


def main():
	labelsFolderPath = sys.argv[1]
	meshes = sys.argv[2]
	labelsPaths = sorted(glob.glob(labelsFolderPath +'*'))
	labelStack = [tifffile.imread(labelsPaths[z]) for z in range(len(labelsPaths))]
	labelStack = np.dstack(labelStack)
	#code.interact(local=locals())
	labels = np.unique(labelStack)[1:]
	
	#code.interact(local=locals())
	count = 0
	for each in labels:
		print(count)
		count += 1
		indices = np.where(labelStack==each)
		print(len(indices[0]))
		#code.interact(local=locals())
		blankImg = np.zeros(labelStack.shape, dtype=np.uint8)
		blankImg[indices] = 1
		
		vertices, normals, faces = march(blankImg.transpose(), 1)  # zero smoothing rounds
		writeObj(meshes + str(each) + '.obj', vertices, normals, faces)
		
	


if __name__ == "__main__":
	main()
