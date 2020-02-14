from tools import dim
import numpy as np
import logging
import sys
import cv2

def main():
    # Get the training data
    with np.load('C:/Master/TTK-4900-Master/data/training_data/200_days_2018/ssl_train.npz', allow_pickle=True) as data:
        ssl_x = data['arr_0'][:,0]
        ssl_y = data['arr_0'][:,1]
    with np.load('C:/Master/TTK-4900-Master/data/training_data/200_days_2018/ssl_train.npz', allow_pickle=True) as data:
        phase_x = data['arr_0'][0][0].flatten()
        phase_y = data['arr_0'][0][1]    
    
        
    # The standard grid size we will use
    gridSize = dim.find_avg_dim(ssl_x)
    nLon = gridSize[0]
    nLat = gridSize[1]
    nTeddies = len(ssl_x)
    
    # Save the resizing till the actual training begins, keep the size differences
    for i in range(nTeddies):
        ssl_x[i][0] = np.array(ssl_x[i][0], dtype='float32') # convert to numpy array
        ssl_x[i][0] = cv2.resize(ssl_x[i][0], dsize=(nLon, nLat), interpolation=cv2.INTER_CUBIC) # Resize to a standard size
        phase_x[i][0] = np.array(phase_x[i][0], dtype='float32') # convert to numpy array
        phase_x[i][0] = cv2.resize(phase_x[i][0], dsize=(nLon, nLat), interpolation=cv2.INTER_CUBIC) # Resize to a standard size
    

    print(dim(ssl_x))


if __name__ == '__main__':
    main()
