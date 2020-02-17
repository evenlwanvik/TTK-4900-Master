import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import cv2
from tools import dim

def getAccuracy(pred_val, true_val):
    correct = 0
    N = len(true_val)
    for i in range(N):
        print(true_val[i])
        print(pred_val[i])
        if true_val[i] == pred_val[i]:
            correct += 1
    return correct/N  


def preprocess_data(data_path):
    with np.load(data_path, allow_pickle=True) as data:
        X = data['arr_0'][:,0]
        Y = data['arr_0'][:,1]
        
    # The standard grid size we will use
    gridSize = dim.find_avg_dim(X)
    nLon = gridSize[0]#+6
    nLat = gridSize[1]#+3
    nTeddies = len(X)

    scaler = MinMaxScaler(feature_range=(-1,1))

    # Save the resizing till the actual training begins, keep the size differences
    # TODO: Make the resizing into a tool? *Being used in training_data as well
    for i in range(nTeddies):
        X[i] = np.array(X[i], dtype='float32') # convert to numpy array
        X[i] = cv2.resize(X[i], dsize=(nLat, nLon), interpolation=cv2.INTER_CUBIC) # Resize to a standard size and flatten
        X[i] = scaler.fit_transform(X[i]) # normalize to [-1,1]

    X = np.array(list(X))

    Y = Y.astype('int') # Apparently it is of type object, make it int
    # return (X_train, X_test, y_train, y_test)
    return train_test_split(X, Y, test_size=0.33)