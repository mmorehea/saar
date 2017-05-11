import code
import tifffile
import numpy as np
import glob
from numpy import load
import sys
import pickle

def main():
	labelsFolderPath = sys.argv[1]
	labelsPaths = sorted(glob.glob(labelsFolderPath +'*'))
	labelStack = [tifffile.imread(labelsPaths[z]) for z in range(len(labelsPaths))]
	labelStack = np.dstack(labelStack)
	#code.interact(local=locals())
	print("stacked")
	mapBegin = {}
	mapEnd = {}
	for ii in range(labelStack.shape[2]):
		print(str(ii))
		img = labelStack[:,:,ii]
		labels = np.unique(img)
		for label in labels:
			if label not in mapBegin:
				mapBegin[label] = ii
			if label in mapEnd:
				if ii > mapEnd[label]:
					mapEnd[label] = ii
			else:
				mapEnd[label] = ii
		#code.interact(local=locals())
	
	listOfGoodLabels = []
	for label in mapBegin:
		if label in mapEnd:
			if mapEnd[label] - mapBegin[label] > 100:
				listOfGoodLabels.append(label)
	

	with open('goodLabels.pickle', 'wb') as fp:
		pickle.dump(listOfGoodLabels, fp)	
				
		
	

		
	


if __name__ == "__main__":
	main()
