import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import cv2
from tools import dim
import matplotlib.pyplot as plt
from collections import Counter


def getAccuracy(pred_val, true_val):
    correct = 0
    N = len(true_val)
    for i in range(N):
        if true_val[i] == pred_val[i]:
            correct += 1
    return correct/N  


def preprocess_data(X, Y, split=True, gridSize=None):

    # The standard grid size we will use
    if gridSize==None:
        gridSize = dim.find_avg_dim(X)
    nLon = gridSize[0]
    nLat = gridSize[1]
    
    nTeddies = len(X)

    print("Resizing {} eddies to ({}, {}) for training".format(nTeddies,nLon,nLat))
    scaler = MinMaxScaler(feature_range=(0,1))

    # Save the resizing till the actual training begins, keep the size differences
    # TODO: Make the resizing into a tool? *Being used in training_data as well
    for i in range(nTeddies):
        X[i] = np.array(X[i], dtype='float32') # convert to numpy array
        X[i] = cv2.resize(X[i], dsize=(nLon, nLat), interpolation=cv2.INTER_CUBIC) # Resize to a standard size and flatten
        X[i] = scaler.fit_transform(X[i]) # normalize to [0,1]

    X = np.array(list(X))

    Y = Y.astype('int') # Apparently it is of type object, make it int
    # return (X_train, X_test, y_train, y_test)
    if split:
        return train_test_split(X, Y, test_size=0.33)
    else:
        return X, Y


def find_dimensions(data_path):

    with np.load(data_path, allow_pickle=True) as data:
        arr = data['arr_0'][:,0]
    shape = arr.shape

    lonDims = [x.shape[0] for x in arr]
    latDims = [x.shape[1] for x in arr]
    
    lonLabels, lonVals = zip(*Counter(lonDims).items())
    latLabels, latVals = zip(*Counter(latDims).items())
    
    lonIdxs = range(lonLabels[-1])
    latIdxs = range(latLabels[-1])

    latVals = [latVals[latLabels.index(i)] if i in latLabels else 0 for i in latIdxs]
    lonVals = [lonVals[lonLabels.index(i)] if i in lonLabels else 0 for i in lonIdxs]

    width = 0.8
    plt.bar(lonIdxs, lonVals, width, alpha=0.5)
    plt.bar(latIdxs, latVals, width, alpha=0.5)
    plt.legend(['Longitude', 'Latitude'])
    plt.xlabel('Sizes')
    plt.ylabel('Sample count')
    plt.title('Distribution of training data sizes')
    plt.show('Length of sample sizes')


def sliding_window(arr, wSize, hSize, windowSize):
    ''' 
    slide window across the image 
    args:
        arr         (float): 2d (image) array
        stepSize    (int): number of pixels to skip per new window
        windowSize  (int): width and height of window in pixels
    returns:
        generator of indexes of windows
    ''' 
    dims = arr.shape

    for y in range(0, dims[1], hSize): 
        for x in range(0, dims[0], wSize):
            # yield current window
            yield x, y, (list(range(x, x+windowSize[0])), list(range(y, y+windowSize[1])))
