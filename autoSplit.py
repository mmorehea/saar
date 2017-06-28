# -*- coding: utf-8 -*-
import argparse
import cv2
import glob
import code
import numpy as np
import sys
from timeit import default_timer as timer
import os
from itertools import cycle
from itertools import combinations
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import cPickle as pickle
import random
import collections
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
from skimage import data
from skimage.feature import match_template
import math
import json
from scipy import ndimage as nd
import tifffile
import re
from __future__ import division
import tifffile
import ConfigParser
from scipy import ndimage as nd
