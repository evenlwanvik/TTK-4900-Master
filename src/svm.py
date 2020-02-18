from tools.machine_learning import getAccuracy, preprocess_data, sliding_window
from tools.sliding_window import localize_and_classify
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from tools.load_nc import load_netcdf4
#from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from tools import gui, dim
import numpy as np
import pickle
import xarray as xr

meastype = 'sst'

data_path = f'C:/Master/TTK-4900-Master/data/training_data/200_days_2018/{meastype}_train.npz'

model_fpath = f'models/svm_{meastype}_01.sav'


def train_model(data_path=data_path, model_fpath=model_fpath):
    
    # Get the training data
    X_train, X_test, y_train, y_test = preprocess_data(data_path)

    y_train, y_test = abs(y_train), abs(y_test) # Single-class

    shape = X_train.shape
    X_train = X_train.reshape(shape[0],shape[1]*shape[2])
    shape = X_test.shape
    X_test = X_test.reshape(shape[0],shape[1]*shape[2])
    
    #pipeline = OneVsRestClassifier(SVC(kernel='rbf', verbose=1, probability=True))
    pipeline = SVC(kernel='rbf', verbose=1, probability=True) # Single-class

    parameters = {
            'gamma': [0.005, 0.01, 0.05, 0.1],
            'C': [0.5, 1, 10],
            'kernel': ['rbf'],
            #'degree': [1, 2, 3, 4, 5],
    }   

    # Create a classifier object with the classifier and parameter candidates
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


def test_model(nc_fpath='C:/Master/data/cmems_data/global_10km/phys_noland_001.nc', model_fpath=model_fpath, meastype=meastype):

    (ds,t,lon,lat,depth,uvel_full,vvel_full,sst_full,ssl_full) =  load_netcdf4(nc_fpath)

    ssl = np.array(ssl_full[0].T, dtype='float32') 
    sst = np.array(sst_full[0].T, dtype='float32') 
    uvel = np.array(uvel_full[0,0].T, dtype='float32') 
    vvel = np.array(vvel_full[0,0].T, dtype='float32') 
    with np.errstate(all='ignore'): # Disable zero div warning
        phase = xr.ufuncs.rad2deg( xr.ufuncs.arctan2(vvel, uvel) ) + 180
    
    ssl_model = pickle.load(open(f'models/svm_ssl_01.sav', 'rb'))
    sst_model = pickle.load(open(f'models/svm_sst_01.sav', 'rb'))
    phase_model = pickle.load(open(f'models/svm_phase_01.sav', 'rb'))

    shape = ssl.shape
    stepSize = 10


    # for every dataset and its model
    for dataIdx, data in enumerate([(ssl, ssl_model), (sst, sst_model), (phase, phase_model),]):
        # loop over the sliding window
        for (x, y, (lonIdxs, latIdxs)) in sliding_window(data, stepSize=stepSize, windowSize=(winW, winH)):

            if lonIdxs[-1] > shape[0] or latIdxs[-1] > shape[1]:
                continue

            lo, la = lon[lonIdxs], lat[latIdxs]

            probabilities = clf.predict_proba( [np.array(window.flatten())] )

            if probabilities[0,1] > probabilities[0,0] and probabilities[0,1] > probLim:
                print(f'data nr {dataIdx} | probas: {probabilities} | x: {x}, y: {y}')
                print(probabilities)
                print(dataIdx)


if __name__ == '__main__':
    #test_model()
    train_model()
