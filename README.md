# saar

## To Do:
- find the meshable labels before the end of the pipeline, remember them, and make meshes for only those.
- superimpose the labels with their colors over the EM for easier proofreading.

## The Pipeline

### 1. GUI that shows the user the top image in the original EM stack and allows him/her to set the thresholding and contouring parameters that will be applied to the entire stack.

#### getParamaters.py
**inputs**: original EM image stack, user input  
**outputs**: config file  
Allows the user to set up the thresholding and contouring parameters using a GUI that displays the top image of the stack, and saves the parameters to a config file.

#### getThreshold.py, mass.py
Outdated scripts that together make up the old version of getParameters.py.

### 2. Generate the labels using the appropriate threshold and contour parameters.

### 3. Make meshes from the labels.

## Other Stuff

## arrayJobCleanLabels, arrayJobContour,arrayJobThreshold
Not sure what these are for, don't appear to be part of the main pipeline

## buildmeshes3.py
**inputs**: path to labels, path to output meshes  
**outputs**: folder full of meshes  
Takes the labels and creates .obj files for each one. multiMesh3.py is more up to date.

## cleanLabels.py
**inputs**: uncleaned labels  
**outputs**: cleaned labels in the folder "clean/"  
Single threaded version of cleaning algorithm, copies each label onto its own blank array to dilate and erode it, then returns to main array.

## clusterSaarCleanLabels.py
**inputs**: uncleaned labels  
**outputs**: cleaned labels in the folder "clean/"  
Multithreaded version of cleanLabels.py.

## clusterSaarContour.py
**inputs**: config file, path to thresholded EM data in folder "thresh"  
**outputs**: contoured images saved to the folder "contour/"  
Using multithreading, finds the contours of each thresholded slice and applies cv2 morphology close to clean them.

## clusterSaarFindLabels.py
**inputs**: cleaned EM image stack in the folder "cleaned/"  
**ouptuts**: labeled EM image stack in hte folder "cleanedLabels32/"  
Takes the cleaned EM stack and applies nd.measurements.label to extract all the labels in the array.

## coolCodes
"command to interactly take over gspirou node"

## deleteLabels.py
**inputs**: path to thresholded EM image stack, list of labels to delete  
**outputs**: image stack with labels removed  
Deletes a list of labels from the stack.

## extractLabels.py
**inputs**: path to thresholded EM image stack, list of labels to extract:  
**outputs**: image stack with only the extracted labels in folder "extractedLabels/"  
Extracts a list of labels from the stack.

## findMissing.py
**inputs**: two paths to different folders  
**outputs**: a list of items present in one folder but not the other  
Determines which items in one folder are missing from another, and returns a list of these items.

## installcv3.sh
bash script for installing cv3.

## marching_cubes.cpython-34m.so
Provides C code for making meshes

## multiMesh3.py
**inputs**: path to labels, path to output meshes  
**outputs**: folder full of meshes  
Multithreaded, updated version of buildMeshes3.py.

## multisaar.py, multisaar2.py, saar.py
Out of date scripts containing an embryonic version of our full pipeline.

## organizeLabels.py
**inputs**: labeled stack of EM images  
**outputs**: folder for each label containing 2 numpy save files, one with the local coordinates of the label and the other with the global coordinates of the bounding box for this label  
Finds the bounding box for each label and changes the global coordinates to local coordinates. Saves each set of local coordinates along with global coordinates for the bounding box of the label.

## parallelMesh3.py
Unsure whether this or multiMesh3 is the more up-to-date.

## singleSaarContour.py, singleSaarThreshold.py
Out of date scripts that use pool.map to perform threshold and contouring.

## tiff_setup.py, tiffile.py
For working with tiffs
