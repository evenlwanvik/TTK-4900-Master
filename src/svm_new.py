from tools.machine_learning import getAccuracy, preprocess_data, sliding_window
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from tools.load_nc import load_netcdf4
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import pickle
import cv2

from sklearn.externals import joblib # To save scaler



sst_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/sst_train.npz'
ssl_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/ssl_train.npz'
uvel_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/uvel_train.npz'
vvel_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/vvel_train.npz'
phase_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/phase_train.npz'
lon_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/lon.npz'
lat_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/lat.npz'
model_fpath = 'D:/master/models/2016/svm_mult_full.h5'
scaler_fpath = "D:/master/models/2016/svm_norm_scaler.pkl"
#2016/new
#200_days_2018

# Create a scaler for each channel
nChannels = 3
scaler = [StandardScaler() for _ in range(nChannels)]
#scaler = MinMaxScaler(feature_range=(-1,1))
winW, winH = int(11), int(6)
probLim = 0.97

def train_model():

    winW2, winH2 = winW*4, winH*4

    # TODO: This part is the same for every model if we use 3 channels, create function
    # Open the numpy training data array and append for each channel we want to use
    X = []
    #with np.load(sst_path, allow_pickle=True) as data:
    #    X.append(data['arr_0'][:,0])    
    with np.load(ssl_path, allow_pickle=True) as data:
        X.append(data['arr_0'][:,0])
        Y = data['arr_0'][:,1]
    with np.load(uvel_path, allow_pickle=True) as data:
        X.append(data['arr_0'][:,0])
    with np.load(vvel_path, allow_pickle=True) as data:
        X.append(data['arr_0'][:,0])

    nTeddies = len(X[0])

    for c in range(nChannels): # For each channel
        for i in range(nTeddies): # For each Training Eddy
            #amin, amax = np.amin(X[c][i]), np.amax(X[c][i])
            #X[c][i] = X[c][i]/90
            X[c][i] = cv2.resize(X[c][i], dsize=(winH2, winW2), interpolation=cv2.INTER_CUBIC) 

    # Reshape data for CNN (sample, width, height, channel)
    X_svm = np.zeros((nTeddies,winW2,winH2,nChannels))
    for i in range(nTeddies): # Eddies
        for lo in range(winW2): # Row
            for la in range(winH2): # Column
                for c in range(nChannels): # Channels
                    X_svm[i,lo,la,c] = X[c][i][lo][la]

    # Create and set the scaler for each channel
    X_svm = X_svm.reshape(nTeddies, -1, nChannels)
    for c in range(nChannels):
        X_svm[:,:,c] = scaler[c].fit_transform(X_svm[:,:,c])
    joblib.dump(scaler, scaler_fpath) # Save the Scaler model

    # flatten each sample for svm, the method should be able to find the non-linear 
    # relationships between the seperate channels anyways.
    X_svm = X_svm.reshape(nTeddies, -1)
    
    for i in range(nTeddies):
        X_svm[i] = X_svm[i].flatten()

    # Want classes to be from 0-2, I've used -1,0,1
    for i in range(len(Y)):
        if Y[i] == -1: Y[i]=2

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_svm, Y[:nTeddies], test_size=0.33)
    nTrain = len(X_train)   

    pipeline = OneVsRestClassifier(SVC(kernel='rbf', verbose=1, probability=True))
    #pipeline = SVC(kernel='rbf', verbose=1, probability=True) # Single-class

    parameters = {
            'estimator__gamma': [0.001, 0.01, 0.1, 1, 10],
            'estimator__C': [0.1, 1, 10, 100],
            #'estimator__kernel': ['rbf', 'poly'],
            #'estimator__degree': [2, 3, 4, 5],
    }       

    # Classifier object with the classifier and parameter candidates for cross-validated grid-search
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=3, scoring="accuracy")

    model.fit(list(X_train), y_train)   

    pickle.dump(model, open(model_fpath, 'wb'))

    y_pred = model.predict(list(X_test))

    accuracy = getAccuracy(y_pred, y_test)
    print(f"> The accuracy of the model is {accuracy}")

    print("Best parameters set found on development set:")
    print(model.best_params_)
    print("Grid scores on development set:")
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))

    print("Detailed classification report:\n")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))

if __name__ == '__main__':
    train_model()
    #test_model()