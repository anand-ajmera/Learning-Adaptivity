#!/usr/bin/python

import numpy as np
import math
# import csv

# with open('../build/features.csv', 'rb') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         print row

NaN = float('nan')
features = np.genfromtxt('../build/features.csv',delimiter=',') #,dtype=float
print len(features),features[0]


cleaned_list = [x for x in features if ~np.isnan(x)] 
	# features.remove(NaN)
print len(cleaned_list), cleaned_list[0]