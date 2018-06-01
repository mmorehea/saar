import numpy as np
import code
import glob
import sys
import math

def createTransformationMatrix(scalingFactor, xTrans, yTrans, zTrans, rotAlpha, rotBeta, rotGamma):
	translationAndScalingMatrix = np.array([[scalingFactor, 0, 0, 0], 
											[0, scalingFactor, 0, 0], 
											[0, 0, scalingFactor, 0], 
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
	
	return newMatrix
	
	
def separateOBJ(pathOfOBJ):
	with open(pathOfOBJ) as f:
		OBJLines = f.readlines()
	
	vertLines = [line.strip() for line in OBJLines if line[0]=='v']
	
	faceLines = [line for line in OBJLines if line[0]=='f']
	
	vertices = np.array([each.split(' ')[1:] for each in vertLines]).astype(float)
	
	# Add a column of ones to make it compatible with transformation matrix
	vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))

	return vertices, faceLines
	
def combinePartsAndSave(vertexMatrix, faceLines, pathToSave):
	OBJLines = ['v ' + ' '.join(each) + '\n' for each in vertexMatrix.astype(str)[:,:-1]]

	OBJLines.extend(faceLines)

	with open(pathToSave, 'w') as f:
		for each in OBJLines:
			f.write(each)

	return None

def main():
	pathToOBJ = sys.argv[1]
	pathToSave = "test1.obj"
	scalingFactor = 8.5
	xTrans, yTrans, zTrans = (20000, 0, 0)
	rotAlpha, rotBeta, rotGamma = (180, 0, 90)
	
	transformationMatrix = createTransformationMatrix(scalingFactor, xTrans, yTrans, zTrans, rotAlpha, rotBeta, rotGamma)
	print(transformationMatrix)
	
	vertices, faceLines = separateOBJ(pathToOBJ)
	
	newVertices = transformVertices(vertices, transformationMatrix)
	
	combinePartsAndSave(newVertices, faceLines, pathToSave)
	

if __name__ == "__main__":
	main()
