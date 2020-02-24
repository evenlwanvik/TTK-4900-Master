from tools.machine_learning import getAccuracy, preprocess_data, sliding_window
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tools.load_nc import load_netcdf4
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from tools import gui, dim
import numpy as np
import pickle
import xarray as xr
import cv2

from skimage.feature import hog
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


meastype = 'phase'
data_path = f'C:/Master/TTK-4900-Master/data/training_data/200_days_2018/{meastype}_train.npz'
model_fpath = f'models/svm_{meastype}_01.sav'

winW, winH = int(15), int(9)

def train_model(data_path=data_path, model_fpath=model_fpath):
    
    # Get the training data
    X_train, X_test, y_train, y_test = preprocess_data(data_path, split=True, gridSize=(winW,winH))

    #y_train, y_test = abs(y_train), abs(y_test) # Single-class

    shape = X_train.shape
    X_train = X_train.reshape(shape[0],shape[1]*shape[2])
    shape = X_test.shape
    X_test = X_test.reshape(shape[0],shape[1]*shape[2])
    
    pipeline = OneVsRestClassifier(SVC(kernel='rbf', verbose=1, probability=True))
    #pipeline = SVC(kernel='rbf', verbose=1, probability=True) # Single-class

    parameters = {
            'estimator__gamma': [0.005, 0.01, 0.05, 0.1, 1],
            'estimator__C': [0.5, 1, 10],
            'estimator__kernel': ['rbf'],
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

    day = 50

    ssl = np.array(ssl_full[day].T, dtype='float32') 
    uvel = np.array(uvel_full[day,0].T, dtype='float32') 
    vvel = np.array(vvel_full[day,0].T, dtype='float32') 
    with np.errstate(all='ignore'): # Disable zero div warning
        phase = xr.ufuncs.rad2deg( xr.ufuncs.arctan2(vvel, uvel) ) + 180
    
    # Recreate the exact same model purely from the file
    ssl_clf = pickle.load(open('models/svm_ssl_01.sav', 'rb'))
    phase_clf = pickle.load(open('models/svm_phase_01.sav', 'rb'))

    shape = ssl.shape
    ssl_probLim = 0.95
    phase_probLim = 0.35
    stepSize = 2
    scaler = MinMaxScaler(feature_range=(0,1))

    # normalize (scale) the data
    ssl_scaled = scaler.fit_transform(ssl)
    phase_scaled = scaler.fit_transform(phase)

    # loop over the sliding window of indeces
    for (x, y, (lonIdxs, latIdxs)) in sliding_window(ssl, stepSize=stepSize, windowSize=(winW, winH)):
        #print('{},{}'.format(x,y))

        #import pdb; pdb.set_trace()

        if lonIdxs[-1] >= shape[0] or latIdxs[-1] >= shape[1]:
            continue

        # Window indexed data
        ssl_wind = np.array([[ssl[i,j] for j in latIdxs] for i in lonIdxs])
        ssl_scaled_wind = np.array([[ssl_scaled[i,j] for j in latIdxs] for i in lonIdxs])
        phase_wind = np.array([[phase[i,j] for j in latIdxs] for i in lonIdxs])
        phase_scaled_wind = np.array([[phase_scaled[i,j] for j in latIdxs] for i in lonIdxs])
        uvel_wind = np.array([[uvel[i,j] for j in latIdxs] for i in lonIdxs])
        vvel_wind = np.array([[vvel[i,j] for j in latIdxs] for i in lonIdxs])

        lo, la = lon[lonIdxs], lat[latIdxs]

        # Predict and receive probability
        ssl_prob   = ssl_clf.predict_proba([ssl_scaled_wind.flatten()])
        phase_prob = phase_clf.predict_proba([phase_scaled_wind.flatten()])

        print(ssl_prob)
        if ssl_prob[0,1] > ssl_prob[0,0] and ssl_prob[0,1] > ssl_probLim:
            fig, ax = plt.subplots(1, 3, figsize=(16, 6))
            print('cyclone | ssl prob: {} | phase prob: {} | lon: [{}, {}, lat: [{}, {}]'.format(ssl_prob[0,1]*100,phase_prob[0,1]*100,lo[0],lo[-1],la[0],la[-1]))
            plot_window(ssl_wind, phase_wind, uvel_wind, vvel_wind, lo, la, ax)

        if ssl_prob[0,2] > ssl_probLim:
            fig, ax = plt.subplots(1, 3, figsize=(16, 6))
            print(phase_prob)
            print('anti-cyclone | ssl prob: {} | phase prob: {} | lon: [{}, {}, lat: [{}, {}]'.format(ssl_prob[0,2]*100,phase_prob[0,2]*100,lo[0],lo[-1],la[0],la[-1]))
            plot_window(ssl_wind, phase_wind, uvel_wind, vvel_wind, lo, la, ax)


def plot_window(ssl, phase, uvel, vvel, lon, lat, ax):
    ax[0].contourf(lon, lat, ssl.T, cmap='rainbow', levels=30)

    n=-uvel
    color_array = np.sqrt(((uvel-n)/2)**2 + ((vvel-n)/2)**2)
    ax[2].quiver(lon, lat, uvel.T, vvel.T, color_array, scale=7) 

    lonNew = np.linspace(lon[0], lon[-1], lon.size*5)
    latNew = np.linspace(lat[0], lat[-1], lat.size*5)

    phase_interp = cv2.resize(phase, dsize=(latNew.size, lonNew.size), interpolation=cv2.INTER_CUBIC)

    levels = MaxNLocator(nbins=10).tick_values(phase.min(), phase.max())
    cmap = plt.get_cmap('CMRmap')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    ax[1].pcolormesh(lonNew, latNew, phase_interp.T, cmap=cmap, norm=norm)

    plt.show() 


def resize_array(a, dSize=None, fx=1, fy=1):
    return cv2.resize(np.array(a, dtype='float32'), dsize=dSize, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

if __name__ == '__main__':
    test_model()
    #train_model()
