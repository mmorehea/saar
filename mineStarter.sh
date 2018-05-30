#! /bin/bash
# restart labelFeatureExtraction.py when it crashes

until killall -9 python; python labelFeatureExtraction.py /media/curie/5TB/saarData/labels_finaltest_flipped_stacked/labels_finaltest_flipped.tif /media/curie/5TB/saarData/labels_finaltest_flipped_stacked/emMended.tif
do
	sleep 1
	echo "Restarting program..."
done
