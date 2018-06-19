# Saar
<p>Saar takes an EM image stack and produces .obj mesh files for axons contained in the volume. An interface is provided for specifying segmentation parameters while previewing the results of each selection.</p>

<h3>If you use Saar, please cite:<h3>

# Requirements
<h3>Software:</h3>
<p>Python 3<br>
Marching Cubes</P>

<h3>Python dependencies:</h3>
<p>SciPy (https://www.scipy.org/)<br>
<p>tifffile.py (https://www.lfd.uci.edu/~gohlke/code/tifffile.py.html)<br>
<p>scikit-image (http://scikit-image.org/docs/dev/api/skimage.html)<br>
<p>Python OpenCV (https://pypi.org/project/opencv-python/)</p>

# Usage

<h3>To run:</h3>
<p>Open a terminal or command window, navigate to the Saar directory, and type:<br><br>
<i>python saar.py emFolder/ emptyFolder1/ emptyfolder2/ destinationFolder/</i><br>
<p>Where 'emFolder' contains a sequence of EM tiff images, the two empty folders will hold the thresholded and labeled images respectively for intermediate steps, and 'destinationFolder' will contain the the resulting meshes.<br><br></p>

<h3>Main Menu</h3>
<p>Saar first presents the user with a main menu, allowing the user to start from any of the main steps in the segmentation process. For the first time, it is recommended that these each be done in order.</p><br>

<h3>Set Parameters</h3>
<p>This step presents the user with a series of trackbars allowing them to set a series of parameters used for segmentation. These trackbars are accompanied by a sample image from the volume allowing the user to preview the effects of changing the parameter. Each parameter is set 3 times, once for each third of the image stack. The parameters include:<br><br>
<b>Threshold Value:</b> Yields a binary image. The regions with intensity above the threshold value are set to 1, and the remaining regions are set to 0. This should be adjusted until the axons are clearly delineated but not too diminished.<br>
<b>Remove Noise Kernel Size:</b> The optimal kernel size for noise removal will vary depending on the image resolution. This should be adjusted to remove the most noise while minimizing distortion to the image.<br>
<b>Filter Size Range:</b> This specifies a percentile range for objects in the image to be removed based on their area in pixels. This should be set to remove the most noise while removing as few axons as possible.<br>
<b>Blob Recovery Radius:</b> This setting applies an algorithm to recover some of the axons that may have been lost during the size filtering step. Usually this can be set to 0, but it may helpful for some image stacks.</p><br>

<h3>Apply Parameters to Whole Stack</h3>
<p>After parameterization, Saar saves the parameters to a file called 'Saar.ini'. This step applies the parameters to each image in the stack, saving the results to the first empty folder specified. The user can then view the stack in image software such as ImageJ and, if needed, adjust the parameters and re-run this step.</p><br>

<h3>Connected Component Labeling</h3>
<p>In this step, Saar labels the image stack according to connected components.</p><br>

<h3>Filter Labels by Size</h3>
<p>Most of the labels produced by connected components are too small to be useful. This step scans the volume and makes a list of the labels that are bigger than a user-specified size threshold. Values in the range of 50-200 are typically effective in reducing to a manageable number of labels. The resulting list of labels is saved in a file called 'outfile.npy'.</p><br>

<h3>Generate Meshes</h3>
<p>This step applies the Marching Cubes algorithm to convert the labels selected in the previous step to a series of meshes with the .obj file extension, allowing them to be viewed in software such as MeshLab and syGlass.</p><br>

<h3>Label Feature Extraction and Classification</h3>
<p>When the meshes from the previous step are examined, the user will notice many false positives (meshes of objects other than axons) as well as some incomplete and falsely merged axons. This step helps remedy this by applying a supervised machine learning algorithm which separates the complete axon meshes from all others. The user must first manually label a few hundred objects in syGlass to provide the training set. The algorithm then performs the following steps:<br>
<ul>
<li>Generation of a csv file containing morphological features of each mesh</li>
<li>Preprocessing of the data using linear discriminant analysis</li>
<li>Creation of support vector machine model using training set</li>
<li>Prediction of labels for the unknown objects</li>
<li>Once the 'good axons' are identified, their obj files are saved to a separate folder</li>
</ul></p><br>

# Output Usage

<p>Saar produces a folder containing a set of .obj mesh files for each axon segmented from the EM volume. These obj files can be used to view the axons in 3D visualization software such as MeshLab and syGlass.</p>
