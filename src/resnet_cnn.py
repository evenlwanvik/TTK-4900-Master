from tools.machine_learning import preprocess_data, sliding_window
from tools import dim
from tools.load_nc import load_netcdf4
import xarray as xr

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import cv2

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Model, load_model, Sequential  
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
import matplotlib.pyplot as plt

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


def identity_block(X, f, filters, stage, block):
    
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X


def convolutional_block(X, f, filters, stage, block, s=2):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape = (64, 64, 3), classes = 6):
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2,2), padding='same')(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


def convert_to_one_hot(y):
    Y_mult = np.array([[ 0 for _ in range(3)] for _ in range(len(y))])
    for i, y in enumerate(y):
        if y==-1: Y_mult[i][1] = 1
        elif y==0: Y_mult[i][0] = 1
        else: Y_mult[i][2] = 1
    return Y_mult

# The mnist network
def mnist(input_shape, nClasses):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.20))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(nClasses, activation='softmax'))
    #model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    # summarize history for accuracy
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
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

def analyse_h5():
    ''' Just to analyze specific training data from the MATLAB GUI '''

    import numpy as np
    import os
    import zipfile
    import scipy.io
    import h5py

    dirpath = 'C:/Master/TTK-4900-Master/data/training_data/2016/h5/'
    zippath = dirpath + 'training_data.zip'

    with zipfile.ZipFile(zippath) as z:
        # read the file
        with z.open('sample_675.h5', 'r') as zf:
            with h5py.File(zf, 'r') as hf:
                label = int(hf['/label'][()])
                lon = hf['/coordinates/lon'][()][0]
                lat = hf['/coordinates/lat'][()][0]
                ssl = hf['/data/ssl'][()]
                ssl = cv2.resize(ssl, dsize=(8, 12), interpolation=cv2.INTER_CUBIC) # Resize to a standard
                plt.contourf(ssl.T, levels=10)
                plt.show()



scaler = MinMaxScaler(feature_range=(-1,1))
nLon, nLat = int(14), int(8)
probLim = 0.9

def train_model():

    sst_path = 'C:/Master/TTK-4900-Master/data/training_data/2016/new/sst_train.npz'
    ssl_path = 'C:/Master/TTK-4900-Master/data/training_data/2016/new/ssl_train.npz'
    uvel_path = 'C:/Master/TTK-4900-Master/data/training_data/2016/new/uvel_train.npz'
    vvel_path = 'C:/Master/TTK-4900-Master/data/training_data/2016/new/vvel_train.npz'
    phase_path = 'C:/Master/TTK-4900-Master/data/training_data/2016/new/phase_train.npz'
    model_fpath = 'models/2016/new/cnn_mult_full.h5'
    #2016/new
    #200_days_2018

    X = []
    #with np.load(sst_path, allow_pickle=True) as data:
    #    X.append(data['arr_0'][:,0])    
    #with np.load(ssl_path, allow_pickle=True) as data:
    #    X.append(data['arr_0'][:,0])
    #    Y = data['arr_0'][:,1]
    with np.load(uvel_path, allow_pickle=True) as data:
        X.append(data['arr_0'][:,0])
    with np.load(vvel_path, allow_pickle=True) as data:
        X.append(data['arr_0'][:,0])
        Y = data['arr_0'][:,1]
    #with np.load(phase_path, allow_pickle=True) as data:
    #    X.append(data['arr_0'][:,0]) 
    #    Y = data['arr_0'][:,1]       
   
    nTeddies = len(X[0])
    nChannels = len(X)

    for c in range(nChannels): # For each channel
        for i in range(nTeddies): # For each Training Eddy
            X[c][i] = cv2.resize(X[c][i], dsize=(nLon, nLat), interpolation=cv2.INTER_CUBIC).T # Resize to a standard size and flatten
            #if c == 4:
            #    X[c][i] = scaler2.fit_transform(X[c][i]) # normalize to [-1,1]
            #else:
            X[c][i] = scaler.fit_transform(X[c][i]) # normalize to [0,1]
            


    X_cnn = np.zeros((nTeddies,nLon,nLat,nChannels))

    for i in range(nTeddies): # Eddies
        for lo in range(nLon): # Row
            for la in range(nLat): # Column
                #X_cnn[i,lo,la,0] = X[0][i][lo][la]
                for c in range(nChannels): # Channels
                    X_cnn[i,lo,la,c] = X[c][i][lo][la]


    X_train, X_test, Y_train, Y_test = train_test_split(X_cnn, Y, test_size=0.33)

    # Adding channel dimension=1
    #X_train = np.expand_dims(np.array(list(X_train)), 3)
    #X_test = np.expand_dims(np.array(list(X_test)), 3)

    model = mnist(input_shape=(nLon, nLat, nChannels), nClasses=3)
    #model = ResNet50(input_shape = (nLon, nLat, nChannels), classes = 3)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    
    # Create 3 columns for each class for multilabel classification
    Y_train = convert_to_one_hot(Y_train)
    Y_test  = convert_to_one_hot(Y_test)
    #for i in range(len(Y_train)):
    #    if Y_train[i] == -1: Y_train[i]=2
    #for i in range(len(Y_test)):
    #    if Y_test[i] == -1: Y_test[i]=2
    

    print('\n\n')
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    print('\n\n')
    model.fit(X_train, Y_train, epochs = 15, batch_size = 1)

    preds = model.evaluate(X_test, Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
    print('\n\n')
    #model.summary()
    #plot_model(model, to_file='ResNet.png')
    #SVG(model_to_dot(model).create(prog='dot', format='svg'))

    y_pred = model.predict(X_test)
    acc = np.equal(np.argmax(Y_test, axis=-1), np.argmax(y_pred, axis=-1)).mean()
    print(acc)

    model.save(model_fpath)


def test_model(nc_fpath='C:/Master/data/cmems_data/global_10km/2016/noland/phys_noland_2016_001.nc'):
    
    print("\n\n")

    model_fpath = 'models/2016/new/cnn_mult_full.h5'

    (ds,t,lon,lat,depth,uvel_full,vvel_full,sst_full,ssl_full) =  load_netcdf4(nc_fpath)

    day = 0

    ssl = np.array(ssl_full[day].T, dtype='float32') 
    uvel = np.array(uvel_full[day,0].T, dtype='float32') 
    vvel = np.array(vvel_full[day,0].T, dtype='float32') 
    with np.errstate(all='ignore'): # Disable zero div warning
        phase = xr.ufuncs.rad2deg( xr.ufuncs.arctan2(vvel, uvel) ) + 180
    
    # Recreate the exact same model purely from the file
    clf = load_model(model_fpath)
    #ssl_clf   = keras.models.load_model('models/2016/cnn_{}class_ssl.h5'.format(cnntype))

    shape = ssl.shape
    stepSize = 4

    print("\n\nperforming sliding window on satellite data \n\n")

    # normalize (scale) the data
    ssl_scaled = scaler.fit_transform(ssl)
    uvel_scaled = scaler.fit_transform(uvel)
    vvel_scaled = scaler.fit_transform(vvel)
    phase_scaled = scaler.fit_transform(phase)

    winW, winH = nLon, nLat
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
        uvel_scaled_wind = np.array([[uvel_scaled[i,j] for j in latIdxs] for i in lonIdxs])
        vvel_wind = np.array([[vvel[i,j] for j in latIdxs] for i in lonIdxs])
        vvel_scaled_wind = np.array([[vvel_scaled[i,j] for j in latIdxs] for i in lonIdxs])

        #channels = [ssl_scaled_wind, uvel_scaled_wind, vvel_scaled_wind, phase_scaled_wind]
        channels = [uvel_scaled_wind, vvel_scaled_wind]
        nChannels = len(channels)
        X_cnn = np.zeros((winW,winH,nChannels))
        for lo in range(winW): # Row
            for la in range(winH): # Column
                #X_cnn[i,lo,la,0] = X[0][i][lo][la]
                for c in range(nChannels): # Channels
                    X_cnn[lo,la,c] = channels[c][lo][la]

        X_cnn = np.expand_dims(X_cnn, 0)

        lo, la = lon[lonIdxs], lat[latIdxs]

        # Predict and receive probability
        prob = clf.predict(X_cnn)

        if prob[0,1] > probLim:
            fig, ax = plt.subplots(1, 3, figsize=(16, 6))
            print('anti-cyclone | prob: {} | lon: [{}, {}] | lat: [{}, {}]'.format(prob[0,1]*100,lo[0],lo[-1],la[0],la[-1]))
            plot_window(ssl_wind, phase_wind, uvel_wind, vvel_wind, lo, la, ax)

        if prob[0,2] > probLim:
            fig, ax = plt.subplots(1, 3, figsize=(16, 6))
            print('cyclone | prob: {} | lon: [{}, {}, lat: [{}, {}]'.format(prob[0,2]*100,lo[0],lo[-1],la[0],la[-1]))
            plot_window(ssl_wind, phase_wind, uvel_wind, vvel_wind, lo, la, ax)

        #if phase_prob[0,0] > phase_probLim:
            #print("phase prob: {} > problim".format(ssl_prob[0,0]))

def plot_window(ssl, phase, uvel, vvel, lon, lat, ax):
    ax[0].contourf(lon, lat, ssl.T, cmap='rainbow', levels=30)

    n=-uvel
    color_array = np.sqrt(((uvel-n)/2)**2 + ((vvel-n)/2)**2)
    ax[2].quiver(lon, lat, uvel.T, vvel.T, color_array, scale=1) 

    levels = MaxNLocator(nbins=10).tick_values(phase.min(), phase.max())
    cmap = plt.get_cmap('CMRmap')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    ax[1].pcolormesh(lon, lat, phase.T, cmap=cmap, norm=norm)

    plt.show() 

if __name__ == '__main__':
    #train_model() 
    #analyse_h5()  
    test_model()