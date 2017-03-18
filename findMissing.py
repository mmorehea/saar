import os
import glob
import sys 

inFolderPath = sys.argv[1]
outFolderPath = sys.argv[2]

inList = [os.path.basename(x) for x in glob.glob(inFolderPath + "*.tif*")]
outList = [os.path.basename(x) for x in glob.glob(outFolderPath + "*.tif*")]


missingList = []
for each in inList:
	if each not in outList:
		missingList.append(each)

print missingList
print len(missingList)
