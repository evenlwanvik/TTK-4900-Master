from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from tools import gui
from tools import dim
import numpy as np
import logging
import pickle
import sys
import cv2
from sklearn.metrics import f1_score

def main():
    # Get the training data
    with np.load('D:/Master/TTK-4900-Master/data/training_data/200_days_2018/ssl_train.npz', allow_pickle=True) as data:
        ssl_x = data['arr_0'][:,0]
        ssl_y = data['arr_0'][:,1]
    with np.load('D:/Master/TTK-4900-Master/data/training_data/200_days_2018/phase_train.npz', allow_pickle=True) as data:  
        phase_x = data['arr_0'][:,0]
        phase_y = data['arr_0'][:,1]
        
    # The standard grid size we will use
    gridSize = dim.find_avg_dim(ssl_x)
    nLon = gridSize[0]#+6
    nLat = gridSize[1]#+3
    nTeddies = len(ssl_x)

    # Save the resizing till the actual training begins, keep the size differences
    # TODO: Make the resizing into a tool? *Being used in training_data as well
    for i in range(nTeddies):
        ssl_x[i] = np.array(ssl_x[i], dtype='float32') # convert to numpy array
        ssl_x[i] = cv2.resize(ssl_x[i], dsize=(nLat, nLon), interpolation=cv2.INTER_CUBIC) # Resize to a standard size and flatten
        phase_x[i] = np.array(phase_x[i], dtype='float32') # convert to numpy array
        phase_x[i] = cv2.resize(phase_x[i], dsize=(nLat, nLon), interpolation=cv2.INTER_CUBIC)

    '''
    feature_descriptor, img_hog = hog(ssl_x[5], orientations=5, pixels_per_cell=(2,2), cells_per_block=(1,1), visualize=True)
    img_hog = np.divide(img_hog, 255.0)
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axs[0,0].imshow(ssl_x[5], cmap='gray')
    axs[0,1].imshow(phase_x[5], cmap='CMRmap')
    #axs[1,0].quiver(None, None, data[2].T, data[3].T, color_array, scale=7)
    axs[1,1].imshow(img_hog, cmap='gray')
    guiEvent, guiValues = gui.show_figure(fig)
    plt.close(fig)
    '''

    ssl_x = np.divide(ssl_x.astype('float').flatten(), 255.0)
    #ssl_y = abs(ssl_y) # just try to detect an eddy in the first round
    ssl_y = ssl_y.astype('int') # Apparently it is of type object, make it int
    #phase_x = np.divide(phase_x, 255.0)
    X_train, X_test, y_train, y_test = train_test_split(ssl_x, ssl_y, test_size=0.33)

    pipeline = OneVsRestClassifier(SVC(kernel='poly'))

    parameters = {
            'estimator__gamma': [0.1, 0.5, 1, 10, 100],
            'estimator__C': [0.1, 1, 10, 100, 1000],
            'estimator__kernel': ['poly', 'rbf'],
            'estimator__degree': [0, 1, 2, 3, 4, 5, 6]
    }   


    # Create a classifier object with the classifier and parameter candidates
    clf_gs = GridSearchCV(pipeline, param_grid=parameters, n_jobs=2, verbose=3, scoring="accuracy")

    clf_gs.fit(X_train, y_train)             

    filename = 'models/svm_01.sav'
    pickle.dump(model, open(filename, 'w+'))

    y_pred = svc_clf.predict(X_test)

    accuracy = getAccuracy(y_pred, y_test)
    print(f"> The accuracy of the model is {accuracy}")


def getAccuracy(pred_val, true_val):
    correct = 0
    N = len(true_val)
    for i in range(N):
        if true_val[i] == pred_val[i]:
            correct += 1
    return correct/N   


if __name__ == '__main__':
    main()
