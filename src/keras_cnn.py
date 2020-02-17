from tools.machine_learning import getAccuracy, preprocess_data
from tools import dim
from tools.load_nc import load_netcdf4
from tools.sliding_window import localize_and_classify
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import numpy as np
import pickle


from keras.models import load_model


data_path = 'C:/Master/TTK-4900-Master/data/training_data/200_days_2018/ssl_train.npz'
model_fpath = 'models/cnn_ssl_01.h5'


def train_model(data_path=data_path, model_fpath=model_fpath):
    
    # Get the training data
    X_train, X_test, y_train, y_test = preprocess_data(data_path)

    y_train = np.array([2 if y==-1 else y for y in y_train])
    y_test = np.array([2 if y==-1 else y for y in y_test])

    X_train = np.expand_dims(np.array(list(X_train)), 3)
    X_test = np.expand_dims(np.array(list(X_test)), 3)
    input_shape = X_train[0].shape

    # Create model
    model = Sequential()
    # Add model layers
    model.add(Conv2D(12, kernel_size=(4,3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(6, kernel_size=(3,2), activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300)


def test_model(nc_fpath='C:/Master/data/cmems_data/global_10km/phys_noland_001.nc', model_fpath=model_fpath):

    (ds,t,lon,lat,depth,uvel_full,vvel_full,sst_full,ssl_full) =  load_netcdf4(nc_fpath)
    ssl = np.array(ssl_full[0].T, dtype='float32') # NC uses (lat,lon), we want (lon/lat) and it needs to be float32

    # load model
    model = load_model(model_fpath)
    # summarize model.
    model.summary()
    #exit()

    localize_and_classify(ssl, model, 'cnn')


if __name__ == '__main__':
    test_model()
    #train_model()
