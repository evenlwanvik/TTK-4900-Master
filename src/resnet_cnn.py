from tools.machine_learning import preprocess_data, sliding_window
from tools import dim
from tools.load_nc import load_nc_data
import xarray as xr

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Turn off tensorflow debugging logs

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Model, load_model, Sequential  
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
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
def mnist(input_shape, classes):
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
    model.add(Dense(classes, activation='softmax'))
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

    dirpath = 'D:/Master/TTK-4900-Master/data/training_data/2016/h5/'
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


def my_model(input_shape, classes):
    # (no of inputs + no of outputs)^0.5 + (1 to 10)
    # sqrt(100)

    model = Sequential()
    model.add(Conv2D(8, kernel_size=(1,1), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(12, kernel_size=(2,2), padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(AveragePooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(24, kernel_size=(2,2), padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(AveragePooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(classes, activation='softmax')) #activation='softmax')?
    return model


##################### TRAIN AND TEST #####################

from sklearn.externals import joblib # To save scaler
from sklearn.preprocessing import StandardScaler


sst_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/sst_train.npz'
ssl_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/ssl_train.npz'
uvel_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/uvel_train.npz'
vvel_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/vvel_train.npz'
phase_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/phase_train.npz'
lon_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/lon.npz'
lat_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/lat.npz'
model_fpath = 'D:/master/models/2016/cnn_mult_full.h5'
scaler_fpath = "D:/master/models/2016/cnn_norm_scaler.pkl"
#2016/new
#200_days_2018


# Create a scaler for each channel
nChannels = 3
scaler = [StandardScaler() for _ in range(nChannels)]
#scaler = MinMaxScaler(feature_range=(-1,1))
winW, winH = int(11), int(6)
probLim = 0.97


# Fortsett å endre oppløsning for å se om vi kan ha mindre, *4 var best
# Også prøv forskjellige kombinasjoner av kanaler


def train_model():

    winW2, winH2 = winW*4, winH*4

    # Open the numpy training data array and append for each channel we want to use
    X = []
    #with np.load(sst_path, allow_pickle=True) as data:
    #    X.append(data['arr_0'][:,0])    
    with np.load(ssl_path, allow_pickle=True) as data:
        X.append(data['arr_0'][:1574,0])
        Y = data['arr_0'][:,1]
    with np.load(uvel_path, allow_pickle=True) as data:
        X.append(data['arr_0'][:1574,0])
    with np.load(vvel_path, allow_pickle=True) as data:
        X.append(data['arr_0'][:1574,0])
    #    Y = data['arr_0'][:,1]
    #with np.load(phase_path, allow_pickle=True) as data:
    #    X.append(data['arr_0'][:,0]) 
    #    Y = data['arr_0'][:,1]       

    # Reshape the "image" to standard size
    nTeddies = len(X[0])
    
    for c in range(nChannels): # For each channel
        for i in range(nTeddies): # For each Training Eddy
            #amin, amax = np.amin(X[c][i]), np.amax(X[c][i])
            #X[c][i] = X[c][i]/90
            X[c][i] = cv2.resize(X[c][i], dsize=(winH2, winW2), interpolation=cv2.INTER_CUBIC) 

    # Reshape data for CNN (sample, width, height, channel)
    X_cnn = np.zeros((nTeddies,winW2,winH2,nChannels))
    for i in range(nTeddies): # Eddies
        for lo in range(winW2): # Row
            for la in range(winH2): # Column
                for c in range(nChannels): # Channels
                    X_cnn[i,lo,la,c] = X[c][i][lo][la]

    
    # Create and set the scaler for each channel
    X_cnn = X_cnn.reshape(nTeddies, -1, nChannels)
    for c in range(nChannels):
        X_cnn[:,:,c] = scaler[c].fit_transform(X_cnn[:,:,c])
    X_cnn = X_cnn.reshape(nTeddies, winW2, winH2, nChannels)
    joblib.dump(scaler, scaler_fpath) # Save the Scaler model

    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X_cnn, Y[:nTeddies], test_size=0.33)
    nTrain = len(X_train)


    '''
    yhot = convert_to_one_hot(Y)
    for i in range(20):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        print(f'\nlabel = {Y[i]}, yhot = {yhot[i]}\n')
        ssl = X_cnn[i,:,:,0] #   X[i,0]
        #ssl_scaled = tmp[i] #   X[i,0]
        
        im = ax.contourf(ssl.T, 30, cmap='rainbow')
        fig.colorbar(im, ax=ax)
        #im2 = ax[1].contourf(ssl_scaled.T, 100, cmap='rainbow')
        #fig.colorbar(im2, ax=ax[1])

        #uvel = X_train[i,:,:,1] #X[i,1]
        #vvel = X_train[i,:,:,2] #X[i,2]
        #n=-1
        #color_array = np.sqrt(((uvel-n)/2)**2 + ((vvel-n)/2)**2)
        #ax.quiver(uvel, vvel, color_array, scale=10)
        plt.show()
    exit()
    '''

    model = mnist(input_shape=(winW2, winH2, nChannels), classes=3)
    #model = ResNet50(input_shape = (winW2, winH2, nChannels), classes = 3)
    #model = my_model(input_shape = (winW2, winH2, nChannels), classes = 3)

    model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
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
    history = model.fit(X_train, Y_train, validation_split=0.33, epochs = 15, batch_size = 1)

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

    plot_history(history)

from cmems_download import download

def test_model(nc_fpath='D:/Master/data/cmems_data/global_10km/2016/noland/phys_noland_2016_060.nc'):
    
    # Download grid to be tested
    latitude = [45, 50]
    longitude = [-24, -12]
    download.download_nc(longitude, latitude)

    # Test cv2 image and sliding window movement on smaller grid
    nc_fpath='D:/Master/data/cmems_data/global_10km/2016/noland/smaller/phys_noland_2016_001.nc'
    
    print("\n\n")

    lon,lat,sst,ssl,uvel,vvel =  load_nc_data(nc_fpath)

    day = 0

    # Create phase if used
    with np.errstate(all='ignore'): # Disable zero div warning
        phase = xr.ufuncs.rad2deg( xr.ufuncs.arctan2(vvel, uvel) ) + 180

    # Recreate the exact same model purely from the file
    clf = load_model(model_fpath)
    #ssl_clf   = keras.models.load_model(D:/master/models/2016/cnn_{}class_ssl.h5'.format(cnntype))

    nLon, nLat = ssl.shape 
    wSize, hSize = 4, 3 

    print("\n\nperforming sliding window on satellite data \n\n")

    # Create canvas to show the cv2 rectangles around predictions
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    # Canvas for zooming in on prediction
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    fig2.show()

    # Draw on the larger canvas before sliding
    ax1.contourf(lon, lat, ssl.T, cmap='rainbow', levels=60)
    n=-1
    color_array = np.sqrt(((uvel-n)/2)**2 + ((vvel-n)/2)**2)
    ax1.quiver(lon, lat, uvel.T, vvel.T, color_array)#, scale=30)#, headwidth=0.5, width=0.01), #units="xy", ) # Plot vector field      
    fig1.subplots_adjust(0,0,1,1)
    fig1.canvas.draw()

    im = np.frombuffer(fig1.canvas.tostring_rgb(), dtype=np.uint8)
    im = im.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
    im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
    imH, imW, _ = im.shape # col, row
    winScaleW, winScaleH = imW/nLon, imH/nLat # Scalar coeff from dataset to cv2 image

    to_be_scaled = [0,1,2] # Only use uvel and vvel
    data = [ssl, uvel, vvel]

    scaler = joblib.load(scaler_fpath) # Import the std sklearn scaler model

    # loop over the sliding window of indeces
    for rectIdx, (x, y, (lonIdxs, latIdxs)) in enumerate(sliding_window(ssl, wSize, hSize, windowSize=(winW, winH))):

        if lonIdxs[-1] >= nLon or latIdxs[-1] >= nLat:
            continue

        winW2, winH2 = winW*4, winH*4
        winSize = (winH2, winW2)

        masked = False # Continue if window hits land

        data_window, data_scaled_window = [], []

        for c in range(len(data)):
            # Creates window, checks if masked, if not returns the window
            a = check_window(data[c], lonIdxs, latIdxs)
            if a is None:
                masked = True
                break

            # append window if not masked
            data_window.append( a )

            # Resize the original window to CNN input dim
            data_window[c] = cv2.resize(data_window[c], dsize=(winSize), interpolation=cv2.INTER_CUBIC)
            if c in to_be_scaled:
                # Create a copy of window to be scaled
                data_scaled_window.append(data_window[c].copy()) 
                i = len(data_scaled_window) - 1
                # Flatten array before applying scalar
                data_scaled_window[i] = data_scaled_window[i].flatten()
                # Scale the data
                data_scaled_window[i] = scaler[i].transform([data_scaled_window[i]])[0]
                # Reshape scaled data to original shape
                data_scaled_window[i] = data_scaled_window[i].reshape(winW2, winH2)
        
        # continue to next window if mask (land) is present
        if masked: continue

        x_, y_ = int(winScaleW*(x)), int(winScaleH*(nLat-y)) # y starts in top left for cv2, want it to be bottom left
        winW_, winH_= int(winScaleW*winW), int(winScaleH*winH)

        X_cnn = np.zeros((1,winW2,winH2,nChannels))
        for lo in range(winW2): # Row
            for la in range(winH2): # Column
                for c in range(nChannels): # Channels
                    X_cnn[0,lo,la,c] = data_scaled_window[c][lo,la]

        # Predict and receive probability
        prob = clf.predict(X_cnn)

        print(f"image {rectIdx}, (x,y) = ({(x, y)}) to ({(x + winW, y + winH)})")

        if any(i >= probLim for i in prob[0,1:]):
            if prob[0,1] >= probLim:
                cv2.rectangle(im, (x_, y_), (x_ + winW_, y_ - winH_), (217, 83, 25), 2) 
                print('anti-cyclone | prob: {}'.format(prob[0,1]*100))
                #print('anti-cyclone | prob: {} | lon: [{}, {}] | lat: [{}, {}]'.format(prob[0,1]*100,lo[0],lo[-1],la[0],la[-1]))
            else:
                cv2.rectangle(im, (x_, y_), (x_ + winW_, y_ - winH_), (0, 76, 217), 2) 
                print('cyclone | prob: {}'.format(prob[0,2]*100))
                #print('cyclone | prob: {} | lon: [{}, {}, lat: [{}, {}]'.format(prob[0,2]*100,lo[0],lo[-1],la[0],la[-1]))
            ax2.clear()
            ax2.contourf( data_window[0].T, cmap='rainbow', levels=20)
            # plot vectors on top of ssl
            n=-1
            color_array = np.sqrt(((data_window[1]-n)/2)**2 + ((data_window[2]-n)/2)**2)
            ax2.quiver(data_window[1].T, data_window[2].T, color_array, scale=3)#, headwidth=0.5, width=0.01), #units="xy", ) # Plot vector field      
            fig2.canvas.draw()

        cv2.imshow("Window", im)
        cv2.waitKey(1)
        #time.sleep(0.05) 
    cv2.waitKey(2)
    cv2.imwrite('D:/master/TTK-4900-Master/images/predicted_grid.png', im)
    

def check_window(data, lonIdxs, latIdxs):
    """ Check if window is masked, if not return array """
    a = np.zeros((len(lonIdxs), len(latIdxs)))
    for i, lo in enumerate(lonIdxs):
        for j, la in enumerate(latIdxs):
            x = data[lo,la]
            if np.ma.is_masked(x):
                return None
            a[i,j] = x
    return a




def plot_window(ssl, phase, uvel, vvel, lon, lat, ax):
    ax[0].contourf(lon, lat, ssl.T, cmap='rainbow', levels=30)

    n=-1
    color_array = np.sqrt(((uvel-n)/2)**2 + ((vvel-n)/2)**2)
    ax[2].quiver(lon, lat, uvel.T, vvel.T, color_array, scale=2) 

    levels = MaxNLocator(nbins=10).tick_values(phase.min(), phase.max())
    cmap = plt.get_cmap('CMRmap')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    ax[1].pcolormesh(lon, lat, phase.T, cmap=cmap, norm=norm)

    plt.show() 


if __name__ == '__main__':
   # train_model() 
    #analyse_h5()  
    test_model()