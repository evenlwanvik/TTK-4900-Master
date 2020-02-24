from tools.machine_learning import preprocess_data, sliding_window
from tools.sliding_window import localize_and_classify
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
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
cnntype = 'mult'

data_path = 'C:/Master/TTK-4900-Master/data/training_data/200_days_2018/{}_train.npz'.format(meastype)
model_fpath = 'models/cnn_{}class_{}_01.h5'.format(cnntype,meastype)

# Average size of training data is (15,9)
winW, winH = int(16), int(10)


def train_model(data_path=data_path, model_fpath=model_fpath):

    # Get the training data
    #X_train, X_test, y_train, y_test = preprocess_data(data_path, split=True)
    X_train, y_train = preprocess_data(data_path, split=False, gridSize=(winW,winH))

    # Create 3 columns for each class for multilabel classification
    y_train_mult = np.array([[ 0 for _ in range(3)] for _ in range(len(y_train))])
    for i, y in enumerate(y_train):
        if y==-1: y_train_mult[i][1] = 1
        elif y==0: y_train_mult[i][0] = 1
        else: y_train_mult[i][2] = 1
    nClasses = 3

    # y_train = abs(y_train) # Single-class

    X_train = np.expand_dims(np.array(list(X_train)), 3)
    input_shape = X_train[0].shape

    model = mnist(input_shape, nClasses)
    #model = create_mutli_label_network(len(X_train), input_shape)
    #model = create_network(len(X_train), input_shape)
    history = model.fit(X_train, y_train_mult, validation_split=0.33, epochs=50)
    
    model.save(model_fpath)

    print(model.evaluate(X_train, y_train_mult))
    plot_history(history)


def create_network(nEddies, input_shape):
    # Create model
    model = Sequential()
    # Add model layers
    keras.Input(shape=(nEddies,), name='2d_array')
    model.add(Conv2D(4, kernel_size=(1), activation='relu', input_shape=input_shape))
    model.add(Conv2D(4, kernel_size=(3), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same')) 
    model.add(Conv2D(4, kernel_size=(1), activation='relu', input_shape=input_shape)) 
    #model.add(Conv2D(2, kernel_size=(3), activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_mutli_label_network(nEddies, input_shape):
    # Create model
    model = Sequential()
    # Add model layers
    keras.Input(shape=(nEddies,), name='2d_array')
    # CONV => RELU => POOL
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(3, 3)))
    # (CONV => RELU) * 1 => POOL
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding="same", input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(3, activation='sigmoid'))   
    # Compile model
    model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_multiple_networks(nNets, nEddies, input_shape):
    # (no of inputs + no of outputs)^0.5 + (1 to 10)
    # sqrt(135)

    model = [0]*nNets
    for i in range(nNets):
        model[i] = Sequential()
        model[i].add(Conv2D(4, kernel_size=(4,3), padding='same', activation='relu', input_shape=input_shape))
        #model[i].add(MaxPool2D())
        if i>=1:
            model[i].add(Conv2D(8, kernel_size=5, padding='same', activation='relu'))
            #model[i].add(MaxPool2D())
        if i>=2:
            model[i].add(Conv2D(12, kernel_size=5, padding='same', activation='relu'))
            #odel[i].add(MaxPool2D(padding='same'))   
        #model[i].add(Flatten())
        #model[i].add(Dense(10, activation='relu'))
        #model[i].add(Dense(1, activation='sigmoid'))
        model[i].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_different_dimensions(data_path=data_path):
    with np.load(data_path, allow_pickle=True) as data:
        X = data['arr_0'][:,0]
    a = list( filter(lambda x: x.shape[0]>50, X) )
    for x in a:
        plt.contourf(x.T, cmap='rainbow', levels=20)
        plt.show()


# The mnist network
def mnist(input_shape, nClasses):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.20))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(nClasses, activation='softmax'))
    model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])
    return model


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


def test_model(nc_fpath='C:/Master/data/cmems_data/global_10km/phys_noland_001_02.nc', model_fpath=model_fpath, meastype=meastype):

    print("\n\n")

    (ds,t,lon,lat,depth,uvel_full,vvel_full,sst_full,ssl_full) =  load_netcdf4(nc_fpath)

    day = 50

    ssl = np.array(ssl_full[day].T, dtype='float32') 
    uvel = np.array(uvel_full[day,0].T, dtype='float32') 
    vvel = np.array(vvel_full[day,0].T, dtype='float32') 
    with np.errstate(all='ignore'): # Disable zero div warning
        phase = xr.ufuncs.rad2deg( xr.ufuncs.arctan2(vvel, uvel) ) + 180
    
    # Recreate the exact same model purely from the file
    ssl_clf   = keras.models.load_model('models/cnn_{}class_ssl_01.h5'.format(cnntype))
    phase_clf = keras.models.load_model('models/cnn_{}class_phase_01.h5'.format(cnntype))

    shape = ssl.shape
    ssl_probLim = 0.95
    phase_probLim = 0.35
    stepSize = 5
    scaler = MinMaxScaler(feature_range=(0,1))

    print("\n\nperforming sliding window on satellite data \n\n")

    # normalize (scale) the data
    ssl_scaled = scaler.fit_transform(ssl)
    phase_scaled = scaler.fit_transform(phase)

    dSize = (winW, winH)

    # loop over the sliding window of indeces
    for (x, y, (lonIdxs, latIdxs)) in sliding_window(ssl, stepSize=stepSize, windowSize=dSize):

        if lonIdxs[-1] >= shape[0] or latIdxs[-1] >= shape[1]:
            continue
        dSize = (winH, winW)
        # Window indexed data and resizing from a smaller window to model size
        ssl_wind = np.array([[ssl[i,j] for j in latIdxs] for i in lonIdxs])
        ssl_scaled_wind = np.array([[ssl_scaled[i,j] for j in latIdxs] for i in lonIdxs])
        phase_wind = np.array([[phase[i,j] for j in latIdxs] for i in lonIdxs])
        phase_scaled_wind = np.array([[phase_scaled[i,j] for j in latIdxs] for i in lonIdxs])
        uvel_wind = np.array([[uvel[i,j] for j in latIdxs] for i in lonIdxs])
        vvel_wind = np.array([[vvel[i,j] for j in latIdxs] for i in lonIdxs])

        # Add dimensions for CNN
        ssl_cnn_window   = np.expand_dims(np.expand_dims(ssl_scaled_wind, 2), 0) # (1, 30, 18, 1)
        phase_cnn_window = np.expand_dims(np.expand_dims(ssl_scaled_wind, 2), 0) # (1, 30, 18, 1)

        lo, la = lon[lonIdxs], lat[latIdxs]

        # Predict and receive probability
        ssl_prob   = ssl_clf.predict_proba(ssl_cnn_window)
        phase_prob = phase_clf.predict_proba(phase_cnn_window)


        if ssl_prob[0,1] > ssl_probLim:
            fig, ax = plt.subplots(1, 3, figsize=(16, 6))
            print('cyclone | ssl prob: {} | phase prob: {} | lon: [{}, {}, lat: [{}, {}]'.format(ssl_prob[0,1]*100,phase_prob[0,1]*100,lo[0],lo[-1],la[0],la[-1]))
            plot_window(ssl_wind, phase_wind, uvel_wind, vvel_wind, lo, la, ax)

        if ssl_prob[0,2] > ssl_probLim:
            fig, ax = plt.subplots(1, 3, figsize=(16, 6))
            print(phase_prob)
            print('anti-cyclone | ssl prob: {} | phase prob: {} | lon: [{}, {}, lat: [{}, {}]'.format(ssl_prob[0,2]*100,phase_prob[0,2]*100,lo[0],lo[-1],la[0],la[-1]))
            plot_window(ssl_wind, phase_wind, uvel_wind, vvel_wind, lo, la, ax)

        #if phase_prob[0,0] > phase_probLim:
            #print("phase prob: {} > problim".format(ssl_prob[0,0]))


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

