import numpy as np
#import csv
import os
#import sys

import scipy.io
import h5py

def h5_open():
    dirpath = 'C:/Master/TTK-4900-Master/data/training_data/2016/h5/'
    filepath = dirpath + 'test.h5'

    with h5py.File(filepath, 'r') as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])

def mat2npz():
    dirpath = 'C:/Master/TTK-4900-Master/data/training_data/2016/mlab/'
    fPath = dirpath + 'test.mat'
    mat = scipy.io.loadmat(fPath)
    print(mat)


def csv2npz():
    dirpath = 'C:/Master/TTK-4900-Master/data/training_data/2016/mlab/'
    #csv.field_size_limit(sys.maxsize) # apparently the csv contains very huge fields

    for fName in os.listdir(dirpath):

        if not fName.endswith(".csv"):
            continue

        fPath = dirpath + fName
        myFile = np.genfromtxt(fPath, delimiter=',')
        print(myFile)
        
        #with open(fPath) as csvfile:

        #    data = list(csv.reader(csvfile))

        #print(data)

        #exit()
        return 0


if __name__ == '__main__':
    h5_open()