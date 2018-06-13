import numpy as np
import code
import glob
import sys
import math

def calculateCentroid(matrix):
	centroid = np.sum(matrix, axis = 0)/matrix.shape[0]
	return centroid

def createTransformationMatrix(xScaling, yScaling, zScaling, xTrans, yTrans, zTrans, rotAlpha, rotBeta, rotGamma):
	translationAndScalingMatrix = np.array([[xScaling, 0, 0, 0], 
											[0, yScaling, 0, 0], 
											[0, 0, zScaling, 0], 
											[xTrans, yTrans, zTrans, 1]])
	rotAlphaMatrix = np.array([[1, 0, 0, 0], 
								[0, math.cos(math.radians(rotAlpha)), -1.0*math.sin(math.radians(rotAlpha)), 0], 
								[0, math.sin(math.radians(rotAlpha)), math.cos(math.radians(rotAlpha)), 0], 
								[0, 0, 0, 1]])
	rotBetaMatrix = np.array([[math.cos(math.radians(rotBeta)), 0, math.sin(math.radians(rotBeta)), 0], 
								[0, 1, 0, 0], 
								[-1.0*math.sin(math.radians(rotBeta)), 0, math.cos(math.radians(rotBeta)), 0], 
								[0, 0, 0, 1]])
	rotGammaMatrix = np.array([[math.cos(math.radians(rotGamma)), -1.0*math.sin(math.radians(rotGamma)), 0, 0], 
								[math.sin(math.radians(rotGamma)), math.cos(math.radians(rotGamma)), 0, 0], 
								[0, 0, 1, 0], 
								[0, 0, 0, 1]])
	rotationMatrix = np.matmul(np.matmul(rotAlphaMatrix, rotBetaMatrix), rotGammaMatrix)
	transformationMatrix = np.matmul(translationAndScalingMatrix,rotationMatrix)
	
	return transformationMatrix

def transformVertices(vertexMatrix, transformationMatrix):
	newMatrix = np.matmul(vertexMatrix,transformationMatrix)
	
	# Move the vertices so that rotation and flipping occurs across the central axis of the mesh rather than the global axis
	# Not working yet
	# xTrans, yTrans, zTrans = (transformationMatrix[3,0],transformationMatrix[3,1],transformationMatrix[3,2])
	# centroid1 = calculateCentroid(vertexMatrix[:,:-1])
	# centroid2 = calculateCentroid(newMatrix[:,:-1]) - np.array([xTrans, yTrans, zTrans])
	# centroid2 = centroid2 / np.array([abs(xScaling), abs(yScaling), abs(zScaling)])
	# realignVector = centroid1 - centroid2
	# newMatrix = np.matmul(newMatrix, np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[realignVector[0],realignVector[1],realignVector[2],1]]))
	
	return newMatrix
	
	
def separateOBJ(pathOfOBJ, invertFaces):
	with open(pathOfOBJ) as f:
		OBJLines = f.readlines()
	
	vertLines = [line.strip() for line in OBJLines if line[0]=='v']
	
	faceLines = [line.strip() for line in OBJLines if line[0]=='f']
	
	vertices = np.array([each.split(' ')[1:] for each in vertLines]).astype(float)
	
	# Add a column of ones to make it compatible with transformation matrix
	vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
	
	faces = np.array([each.split(' ')[1:] for each in faceLines]).astype(float)
	
	# Invert the vertices if the scaling was negative so the mesh looks nice
	faces = faces[:,::int(invertFaces)]
	
	return vertices, faces
	
def combinePartsAndSave(vertexMatrix, faces, pathToSave):
	OBJLines = ['v ' + ' '.join(each) + '\n' for each in vertexMatrix.astype(str)[:,:-1]]
	faceLines = ['f ' + ' '.join(each) + '\n' for each in faces.astype(str)]
	
	OBJLines.extend(faceLines)

	with open(pathToSave, 'w') as f:
		for each in OBJLines:
			f.write(each)

	return None

def main():
	meshesPaths = glob.glob('goodAxons/goodAxonsFromClassification/*.obj') + glob.glob('goodAxons/goodAxonsFromMining/*.obj')
	outDir = 'goodAxons_aligned2/'
	# meshesPaths = glob.glob('deleteThis1/*.obj')
	# outDir = 'deleteThis2/'
	xScaling, yScaling, zScaling = (10.0, 10.0, 5.45454)
	xTrans, yTrans, zTrans = (545.0, 545.0, 0.0)
	rotAlpha, rotBeta, rotGamma = (0.0, 0.0, 0.0)
	invertFaces = xScaling * yScaling * zScaling / abs(xScaling * yScaling * zScaling)
	
	for i, pathToOBJ in enumerate(meshesPaths):
		print(str(i+1) + '/' + str(len(meshesPaths)))
		
		transformationMatrix = createTransformationMatrix(xScaling, yScaling, zScaling, xTrans, yTrans, zTrans, rotAlpha, rotBeta, rotGamma)
		print(transformationMatrix)
		
		vertices, faces = separateOBJ(pathToOBJ, invertFaces)
		
		newVertices = transformVertices(vertices, transformationMatrix)
		
		combinePartsAndSave(newVertices, faces, outDir + pathToOBJ.split('/')[-1])
	

if __name__ == "__main__":
	main()
