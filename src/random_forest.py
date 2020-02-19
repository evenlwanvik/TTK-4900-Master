from tools.machine_learning import getAccuracy, preprocess_data, sliding_window
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tools.load_nc import load_netcdf4
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from tools import dim
import numpy as np
import sys
import pickle
import xarray as xr
import cv2


from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


meastype = 'ssl'
data_path = f'C:/Master/TTK-4900-Master/data/training_data/200_days_2018/{meastype}_train.npz'
model_fpath = f'models/rf_{meastype}_01.sav'

winW, winH = int(15*2), int(9*2)

def train_model(data_path=data_path, model_fpath=model_fpath):


    # Get the training data
    X_train, X_test, y_train, y_test = preprocess_data(data_path)

    y_train, y_test = abs(y_train), abs(y_test) # Single-class

    shape = X_train.shape
    X_train = X_train.reshape(shape[0],shape[1]*shape[2])
    shape = X_test.shape
    X_test = X_test.reshape(shape[0],shape[1]*shape[2])

    # Create the model with 100 trees
    model = RandomForestClassifier(n_estimators=100, 
                                bootstrap = True,
                                max_features = 'sqrt',
                                verbose=1, n_jobs=4)

    model.fit(list(X_train), y_train)      

    pickle.dump(model, open(model_fpath, 'wb'))

    y_pred = model.predict(list(X_test))

    accuracy = getAccuracy(y_pred, y_test)
    print(f"> The accuracy of the model is {accuracy}")

    print(classification_report(y_test, y_pred))




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
    ssl_probLim = 0.89
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

        lo, la = lon[lonIdxs], lat[latIdxs]

        # Predict and receive probability
        ssl_prob   = ssl_clf.predict_proba([ssl_scaled_wind.flatten()])
        phase_prob = phase_clf.predict_proba([phase_scaled_wind.flatten()])

        if ssl_prob[0,1] > ssl_prob[0,0] and ssl_prob[0,1] > ssl_probLim:
            
            #print("ssl prob: {} > problim".format(ssl_prob[0,0]))
            #if phase_prob[0,0] > phase_probLim:
            print('Predicted class {} | ssl prob: {} | phase prob: {} | lon: [{}, {}, lat: [{}, {}]'.format(ssl_clf.predict([phase_scaled_wind.flatten()]),ssl_prob[0],phase_prob[0],lo[0],lo[-1],la[0],la[-1]))

            fig, ax = plt.subplots(1, 2, figsize=(14, 8))

            ax[0].contourf(lo, la, ssl_wind.T, cmap='rainbow', levels=20)

            plot_phase(phase_wind, lo, la, ax[1])

            plt.show() 

        #if phase_prob[0,0] > phase_probLim:
            #print("phase prob: {} > problim".format(ssl_prob[0,0]))


def plot_phase(phase, lon, lat, ax):
    ''' Interpolate phase to make it easier to visualize the eddy center '''
    lonNew = np.linspace(lon[0], lon[-1], lon.size*5)
    latNew = np.linspace(lat[0], lat[-1], lat.size*5)

    phase_interp = cv2.resize(phase, dsize=(latNew.size, lonNew.size), interpolation=cv2.INTER_CUBIC)

    levels = MaxNLocator(nbins=10).tick_values(phase.min(), phase.max())
    cmap = plt.get_cmap('CMRmap')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    ax.pcolormesh(lonNew, latNew, phase_interp.T, cmap=cmap, norm=norm)


if __name__ == '__main__':
    test_model()
    #train_model()
