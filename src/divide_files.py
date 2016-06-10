#!/usr/bin/python

import glob
import numpy as np
from shutil import copyfile
import re
import os

data_folder = "../At_Home"
train_folder = "../training_samples"
test_folder = "../test_samples"
train_size = 20
obj_dir = np.array(glob.glob(data_folder + '/*'))
count = 0


for obj in obj_dir:
    num = 0

    # print "Obj is " , obj
    files = np.array(glob.glob(obj + '/*'))

    for f in files:
        # print "File f is " ,f
        m = re.search('At_Home/(.+?).pcd',f)
        # print "m is ", m
        if m:
            # [int(s) for s in m.group(1).split() if s.isdigit()]
            t = re.search('(.+?)/',m.group(1))

            # n_1 = re.search('/(\d)',m.group(1))
            # n_2 = re.search('/(\d\d)',m.group(1))

            # if n_2:
            #     file_name =  t.group(1) + '_' + n_2.group(1)
            # elif n_1:
            #     file_name =  t.group(1) + '_' + n_1.group(1)

            if t:
                if (count == train_size):
                    num = 0
                    count += 1
                file_name = t.group(1) + '_' + str(num)
                num += 1

        if(count < train_size):
            dirname = train_folder + '/' + t.group(1) + '/' + file_name
            # print "Dir name is " , dirname
            if not os.path.exists(os.path.dirname(dirname)):
                try:
                    os.makedirs(os.path.dirname(dirname))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise


                # print "file name is " , file_name 
            copyfile(f,dirname+'.pcd') #'/'+file_name+
            count += 1

        else:

        #     dirname = test_folder + '/' + t
        #     if not os.path.exists(os.path.dirname(dirname)):
        #     try:
        #         os.makedirs(os.path.dirname(dirname))
        #     except OSError as exc: # Guard against race condition
        #         if exc.errno != errno.EEXIST:
        #             raise

            copyfile(f,test_folder+'/'+file_name+'.pcd')
            # count += 1

    count = 0        		



