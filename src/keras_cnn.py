from tools.machine_learning import preprocess_data, sliding_window
from tools.sliding_window import localize_and_classify
from keras.layers import Dense, Conv2D, Flatten
from sklearn.preprocessing import MinMaxScaler
from tools.load_nc import load_netcdf4
from keras.models import Sequential
from keras.models import load_model
from tensorflow import keras
import matplotlib.pyplot as plt
from tools import dim
import numpy as np
import xarray as xr
import pickle
import cv2



from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

meastype = 'phase'
cnntype = 'bin'
data_path = 'C:/Master/TTK-4900-Master/data/training_data/200_days_2018/{}_train.npz'.format(meastype)
model_fpath = 'models/cnn_{}class_{}_01.h5'.format(cnntype,meastype)

# Average size of training data is (15,9)
winW, winH = int(15*1.5), int(9*1.5)

def train_model(data_path=data_path, model_fpath=model_fpath):
    
    # Get the training data
    #X_train, X_test, y_train, y_test = preprocess_data(data_path, split=True)
    X_train, y_train = preprocess_data(data_path, split=False, gridSize=(winW,winH))

    y_train = abs(y_train) # Single-class

    X_train = np.expand_dims(np.array(list(X_train)), 3)
    input_shape = X_train[0].shape

    # Create model
    model = Sequential()
    # Add model layers
    keras.Input(shape=(len(X_train),), name='digits')
    model.add(Conv2D(12, kernel_size=(4,3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(6, kernel_size=(3,2), activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model
    history = model.fit(X_train, y_train, validation_split=0.33, epochs=200)

    # Save the model
    model.save(model_fpath)

    plot_history(history)


def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    # summarize history for accuracy
    ax[0].plot(history.history['acc'])
    ax[0].plot(history.history['val_acc'])
    ax[0].set_title('model accuracy')
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('model loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['train', 'test'], loc='upper left')
    plt.show()


def test_model(nc_fpath='C:/Master/data/cmems_data/global_10km/phys_noland_001.nc', model_fpath=model_fpath, meastype=meastype):

    print("\n\n")

    (ds,t,lon,lat,depth,uvel_full,vvel_full,sst_full,ssl_full) =  load_netcdf4(nc_fpath)

    day = 50

    ssl = np.array(ssl_full[day].T, dtype='float32') 
    uvel = np.array(uvel_full[day,0].T, dtype='float32') 
    vvel = np.array(vvel_full[day,0].T, dtype='float32') 
    with np.errstate(all='ignore'): # Disable zero div warning
        phase = xr.ufuncs.rad2deg( xr.ufuncs.arctan2(vvel, uvel) ) + 180
    
    # Recreate the exact same model purely from the file
    ssl_clf   = keras.models.load_model('models/cnn_binclass_ssl_01.h5')
    phase_clf = keras.models.load_model('models/cnn_binclass_phase_01.h5')

    shape = ssl.shape
    ssl_probLim = 0.96
    phase_probLim = 0.35
    stepSize = 5
    scaler = MinMaxScaler(feature_range=(0,1))

    print("\n\nperforming sliding window on satellite data \n\n")

    # for every dataset and its model
    #for dataIdx, (data, clf) in enumerate([(ssl, ssl_model), (phase, phase_model),]):

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

        # Add dimensions for CNN
        ssl_cnn_window   = np.expand_dims(np.expand_dims(ssl_scaled_wind, 2), 0) # (1, 30, 18, 1)
        phase_cnn_window = np.expand_dims(np.expand_dims(ssl_scaled_wind, 2), 0) # (1, 30, 18, 1)

        lo, la = lon[lonIdxs], lat[latIdxs]

        # Predict and receive probability
        ssl_prob   = ssl_clf.predict_proba(ssl_cnn_window)
        phase_prob = phase_clf.predict_proba(phase_cnn_window)

        if ssl_prob[0,0] > ssl_probLim:
            #print("ssl prob: {} > problim".format(ssl_prob[0,0]))
            #if phase_prob[0,0] > phase_probLim:
            print('ssl prob: {} | phase prob: {} | lon: [{}, {}, lat: [{}, {}]'.format(ssl_prob[0,0],phase_prob[0,0],lo[0],lo[-1],la[0],la[-1]))

            fig, ax = plt.subplots(1, 3, figsize=(16, 6))

            ax[0].contourf(lo, la, ssl_wind.T, cmap='rainbow', levels=20)

            plot_phase(phase_wind, lo, la, ax[1])

            uvel_wind = np.array([[uvel[i,j] for j in latIdxs] for i in lonIdxs])
            vvel_wind = np.array([[vvel[i,j] for j in latIdxs] for i in lonIdxs])
            n=-1
            color_array = np.sqrt(((uvel_wind-n)/2)**2 + ((vvel_wind-n)/2)**2)
            ax[2].quiver(lo, la, uvel_wind.T, vvel_wind.T, color_array, scale=7) 

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
    #test_model()
    train_model()
