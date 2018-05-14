#!/bin/bash
# restart labelFeatureExtraction.py when it crashes

while [ 1 ]
do uptime
killall python labelFeatureExtraction.py /media/curie/5TB/saarData/labels_finaltest_stacked/labels_finaltest.tif /media/curie/5TB/saarData/labels_finaltest_stacked/emMended.tif
psg httpd | wc
sleep 30
done
