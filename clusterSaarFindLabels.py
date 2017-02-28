



def main():
	threshPath = sys.argv[1]
	threshPaths = sorted(glob.glob(threshPath +'*'))
	outputFolderPath = sys.argv[2]
	
	emImages = [cv2.imread(threshPaths[z], -1) for z in xrange(len(threshPaths))]
	blank = np.dstack(emImages)
	
	labels = nd.measurements.label(blank)
	labels = labels[0]
	
	for each in xrange(labels.shape[2]):
		print each
		#code.interact(local=locals())
		img = labels[:,:,each]
		tifffile.imsave('labels/' + str(each).zfill(4) + '.tif', img)



cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
