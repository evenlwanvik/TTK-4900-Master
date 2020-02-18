from sklearn.ensemble import RandomForestClassifier
from tools.sliding_window import localize_and_classify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tools.load_nc import load_netcdf4
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from tools import dim
from tools.machine_learning import accuracy
import numpy as np
import pickle
import sys
import cv2


data_path = 'C:/Master/TTK-4900-Master/data/training_data/200_days_2018/phase_train.npz'

model_fpath = 'models/rf_phase_01.sav'

def train_model(data_path=data_path, model_fpath=model_fpath):
    # Get the training data
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
        X[i] = cv2.resize(X[i], dsize=(nLat, nLon), interpolation=cv2.INTER_CUBIC) # Resize to a standard size 
        X[i] = list(scaler.fit_transform(X[i]).flatten()) # Scale and flatten


    Y = Y.astype('int') # Apparently my labeling is of type object, make it int
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

    # Create the model with 100 trees
    model = RandomForestClassifier(n_estimators=100, 
                                bootstrap = True,
                                max_features = 'sqrt',
                                verbose=1)

    model.fit(list(X_train), y_train)             

    pickle.dump(model, open(model_fpath, 'wb'))

    y_pred = model.predict(list(X_test))

    accuracy = getAccuracy(y_pred, y_test)
    print(f"> The accuracy of the model is {accuracy}")








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













def test_model(nc_fpath='C:/Master/data/cmems_data/global_10km/phys_noland_001.nc', model_fpath=model_fpath):

    (ds,t,lon,lat,depth,uvel_full,vvel_full,sst_full,ssl_full) =  load_netcdf4(nc_fpath)
    ssl = np.array(sst_full[0].T, dtype='float32') # NC uses (lat,lon), we want (lon/lat) and it needs to be float32
    loaded_model = pickle.load(open(model_fpath, 'rb'))

    localize_and_classify(ssl, lon, lat, loaded_model, 'svm', winW=15, winH=9)#, draw_window=True) 

if __name__ == '__main__':
    #test_model()
    train_model()
