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


def calcMesh(label, labelStack):
	
	indices = np.where(labelStack==label)
	if len(indices[0]) < 5000:
		print("Too small", str(len(indices[0])))
		return
	blankImg = np.zeros(labelStack.shape, dtype=np.uint8)
	blankImg[indices] = 1
	vertices, normals, faces = march(blankImg.transpose(), 1)  # zero smoothing rounds
	with open("multimesh/" + str(label)+".obj", 'w') as f:
		f.write("# OBJ file\n")
		for v in vertices:
			f.write("v %.2f %.2f %.2f \n" % (v[0], v[1], v[2]))
		for n in normals:
			f.write("vn %.2f %.2f %.2f \n" % (n[0], n[1], n[2]))
		for face in faces:
			f.write("f %d %d %d \n" % (face[0]+1, face[1]+1, face[2]+1))

# The worker thread pulls an item from the queue and processes it
def worker(q, labelStack):
	while True:
		item = q.get()
		print('Processing job:', item)
		calcMesh(item, labelStack)
		q.task_done()

def main():
	q = queue.Queue()
	
	
	
	labelsFolderPath = "outMended/"
	labelsPaths = sorted(glob.glob(labelsFolderPath +'*'))
	labelStack = [tifffile.imread(labelsPaths[z]) for z in range(len(labelsPaths))]
	labelStack = np.dstack(labelStack)
	print("Loaded data...")
	labels = np.unique(labelStack)[1:]
	print("Found labels...")
	print("Number of labels", str(len(labels)))
	for label in labels:
		q.put(label)

	for i in range(7):
		     t = threading.Thread(target=worker, args = (q, labelStack,))
		     t.daemon = True  # thread dies when main thread (only non-daemon thread) exits.
		     t.start()
	q.join()
	
		

		
		
	


if __name__ == "__main__":
	main()
