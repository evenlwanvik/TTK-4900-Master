from tools.machine_learning import sliding_window
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import BoundaryNorm
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from matplotlib.ticker import MaxNLocator
from tools.load_nc import load_nc_phys
from sklearn.externals import joblib # To save scaler
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt
from tools import dim
import xarray as xr
import numpy as np
from tools import cnn_models
import cv2
import time
import os

import tensorflow as tf

'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
'''
##################### TOOLS #####################

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

def analyse_h5():
    ''' Just to analyze specific training data from the MATLAB GUI '''

    import zipfile
    import scipy.io
    import h5py

    dirpath = 'C:/Users/47415/Master/TTK-4900-Master/data/h5/'
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

def convert_to_one_hot(y):
    Y_mult = np.array([[ 0 for _ in range(3)] for _ in range(len(y))])
    for i, y in enumerate(y):
        if y==-1: Y_mult[i][1] = 1
        elif y==0: Y_mult[i][0] = 1
        else: Y_mult[i][2] = 1
    return Y_mult

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

def same_dist_elems(arr):
    diff = arr[1] - arr[0]
    for x in range(1, len(arr) - 1):
        if arr[x + 1] - arr[x] != diff:
            return False
    return True

def draw_rectangles(image, rectangles, x, y, winScaleW, winScaleH, eddytype='cyclone'):
    """Draw rectangles on cv2 image"""
    if eddytype=='cyclone': color = (0, 76, 217)
    else:                   color = (217, 83, 25)
    for r in rectangles:
        cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), color, 2)
        ctr = ( int( (r[0]+(r[2]-r[0])/2)/winScaleW ), int( (r[1]+(r[3]-r[1])/2)/winScaleH ) )
        if x.ndim > 1: textLabel = "{} ({:.2f},{:.2f})".format(eddytype,x[ctr],y[ctr])
        else:          textLabel = "{} ({:.2f},{:.2f})".format(eddytype,x[ctr[0]],y[ctr[1]])
        (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,0.5,1)
        textOrg = (r[0], r[1])#+baseLine+15)
        cv2.rectangle(image, (textOrg[0] - 3, textOrg[1]+baseLine - 3), (textOrg[0]+retval[0] + 3, textOrg[1]-retval[1] - 3), (0, 0, 0), 2)
        cv2.rectangle(image, (textOrg[0] - 3, textOrg[1]+baseLine - 3), (textOrg[0]+retval[0] + 3, textOrg[1]-retval[1] - 3), (255, 255, 255), -1)
        cv2.putText(image, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

def draw_simple_rectangle(image, rectangles, eddytype='cyclone'):
    if eddytype=='cyclone': color = (0, 76, 217)
    else:                   color = (217, 83, 25)
    for r in rectangles:
        cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), color, 2)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

##################### WRAPPERS FOR TRAINING #####################

def find_average_model(nSamples=None):
    """ Returns the average and variance of the models """

    train_acc = []
    test_acc = []
    val_acc = []

    N = 15
    for i in range(N):

        x, y, z = train_model(nSamples=nSamples)

        train_acc.append(x)
        val_acc.append(y)
        test_acc.append(z)

    return np.average(train_acc), np.average(val_acc), np.average(test_acc)

def find_best_model():
    """ Returns the best performing model """
    accuracy, accuracy_class, f1_class, precision_class, recall_class, model = 0, 0, 0, 0, 0, None
    for i in range(15):
        accuracy_, accuracy_class_, f1_class_, precision_class_, recall_class_, model_, _ = train_model()
        if accuracy_ > accuracy:
            accuracy = accuracy_
            accuracy_class = accuracy_class_
            f1_class = f1_class_
            precision_class = precision_class_
            recall_class = recall_class_
            model = model_


    print("acc: " + str(accuracy))
    print("Acc: " + str(accuracy_class))
    print("f1_score: " + str(f1_class))
    print("precision: " + str(precision_class))
    print("recall: " + str(recall_class))

    model.save("cnn_model.h5")

def find_best_input_size(sizes=[40]):
    """ Returns the average and variance of the models """

    accuracies = []
    accuracy = []
    t = []
    sigma = []
    time = []
    #sizes = np.arange(5, 80, 5)

    for size in sizes:
    #for size in [80]:
        accuracy = []
        N = 20
        for j in range(N):
            tf.keras.backend.clear_session
            accuracy_, _, _, _, _, _, t_ = train_model(size=size)
            accuracy.append(accuracy_)
            t.append(t_)

        time.append(np.average(t))
        accuracies.append(np.average(accuracy))
        sigma.append(np.std(accuracy))
        print("Average accuracy: " + str(np.average(accuracy)))
        print("Standard deviation: " + str(np.std(accuracy)))

    return accuracies, sigma, time


##################### TRAIN AND TEST #####################

# Hardcoded path to training data, need to fix this
sst_path = 'C:/Users/47415/Master/TTK-4900-Master/data/sst_train.npz'
ssl_path = 'C:/Users/47415/Master/TTK-4900-Master/data/ssl_train.npz'
uvel_path = 'C:/Users/47415/Master/TTK-4900-Master/data/uvel_train.npz'
vvel_path = 'C:/Users/47415/Master/TTK-4900-Master/data/vvel_train.npz'
phase_path = 'C:/Users/47415/Master/TTK-4900-Master/data/phase_train.npz'
lon_path = 'C:/Users/47415/Master/TTK-4900-Master/data/lon.npz'
lat_path = 'C:/Users/47415/Master/TTK-4900-Master/data/lat.npz'
model_fpath = 'C:/Users/47415/Master/TTK-4900-Master/models/cnn_model.h5'
#model_fpath = 'C:/Users/47415/Master/TTK-4900-Master/models/cnn_ssl.h5'
scaler_fpath = "C:/Users/47415/Master/TTK-4900-Master/models/cnn_norm_scaler.pkl"

# Create a scaler for each channel
nChannels = 2
scaler = [StandardScaler() for _ in range(nChannels)]
#scaler = [MinMaxScaler(feature_range=(0,1)) for _ in range(nChannels)]
winW, winH = int(11), int(6)


def train_model(size=40, nSamples=None,use_existing_split=False, trainsplit_dir='C:/Users/47415/Master/TTK-4900-Master/data/train_test_split/'):
    """ Train CNN model, 
    Size:
    'trainsplit_dir' is path to either existing split or where the new split is to be stored, 
    'use_existing_split' flags if an existing split is to be used """

    #winW2, winH2 = winW*6, winH*6
    winW2, winH2 = size, size # Testing a model

    if not use_existing_split:
        # Open the numpy training data array and append for each channel we want to use
        X = []
        with np.load(uvel_path, allow_pickle=True) as data:
            # Random sample indexes to be used
            if nSamples is not None:
                idxs = np.arange(nSamples)
                #idxs = np.random.choice(len(data['arr_0']), nSamples)
            else:
                idxs = np.arange(len(data['arr_0']))
            data = data['arr_0'][idxs]
            X.append(data[:,0])
            Y = data[:,1]
        with np.load(vvel_path, allow_pickle=True) as data:
            data = data['arr_0'][idxs]
            X.append(data[:,0])
         

        # Reshape the "image" to standard size
        nTeddies = len(X[0])

        for c in range(nChannels): # For each channel
            for i in range(nTeddies): # For each Training Eddy
                X[c][i] = np.array(X[c][i], dtype='float32')
                X[c][i][np.isnan(X[c][i])] = 0 # If we have land present, just set to zero TODO: change it?
                X[c][i] = cv2.resize(X[c][i], dsize=(winH2, winW2), interpolation=cv2.INTER_CUBIC) 

        # Reshape data (sample, width, height, channel) for CNN
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

        joblib.dump(scaler, scaler_fpath)
        #exit()

        # Train/test split
        X_train, X_test, Y_train, Y_test = train_test_split(X_cnn, Y[:nTeddies], test_size=0.33)
        nTrain = len(X_train)

        # Create 3 columns for each class for multilabel classification
        Y_train = convert_to_one_hot(Y_train)
        Y_test  = convert_to_one_hot(Y_test)

        np.savez_compressed( trainsplit_dir + 'X_train.npz', X_train)
        np.savez_compressed( trainsplit_dir + 'X_test.npz', X_test)
        np.savez_compressed( trainsplit_dir + 'Y_train.npz', Y_train)
        np.savez_compressed( trainsplit_dir + 'Y_test.npz', Y_test)

        input_shape = (winW2, winH2, nChannels)
        model = cnn_models.my_VGG(input_shape=input_shape,classes=3)
        #model = cnn_models.my_resnet(input_shape=input_shape,classes=3)
        #model = cnn_models.my_inception(input_shape=input_shape,classes=3)

        model.compile(optimizer='adagrad', loss='categorical_crossentropy',  metrics=['acc',f1_m,precision_m, recall_m]) # USE THIS
        #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

        #history = model.fit(X_train, Y_train, validation_split=0.33, epochs = 25, batch_size = 1, callbacks=callbacks_list)
        #history = model.fit(X_train, Y_train, validation_split=0.33, epochs=20, batch_size=10) # USE THIS
        
        from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
        callbacks_list = [earlyStopping, mcp_save, reduce_lr_loss]

        t0 = time.time()
        history = model.fit(X_train, Y_train, epochs=40, batch_size = 10, validation_split=0.33, callbacks=callbacks_list)
        t1 = time.time()

    else:
        custom_objects  = {
            "f1_m": f1_m,
            "precision_m": precision_m, 
            "recall_m": recall_m
        }

        model = load_model(model_fpath, custom_objects=custom_objects)
        
        with np.load( trainsplit_dir + 'X_train.npz', allow_pickle=True ) as h5f:
            X_train = h5f['arr_0']
        with np.load( trainsplit_dir + 'X_test.npz', allow_pickle=True ) as h5f:
            X_test = h5f['arr_0']
        with np.load( trainsplit_dir + 'Y_train.npz', allow_pickle=True ) as h5f:
            Y_train = h5f['arr_0']
        with np.load( trainsplit_dir + 'Y_test.npz', allow_pickle=True ) as h5f:
            Y_test = h5f['arr_0']

    print('\n\n')
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    print('\n\n')


    preds = model.evaluate(X_test, Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
    print('\n\n')

    Y_pred = model.predict(X_test)
    acc = np.equal(np.argmax(Y_test, axis=-1), np.argmax(Y_pred, axis=-1)).mean()

    # store model
    model.save(model_fpath)

    # evaluate the model
    loss, accuracy, f1_keras, precision, recall = model.evaluate(X_test, Y_test, verbose=1)

    print ("Loss: " + str(loss))
    print ("Acc: " + str(accuracy))
    print ("f1_score: " + str(f1_keras))
    print ("precision: " + str(precision))
    print ("recall: " + str(recall))

    print('\n\n')

    # Create 1d array of predicted class
    Y_pred = np.argmax(Y_pred, axis=-1)
    Y_test = np.argmax(Y_test, axis=-1)

    accuracy_class = accuracy_score(Y_test, Y_pred)
    f1_class = f1_score(Y_test, Y_pred, average=None)
    precision_class = precision_score(Y_test, Y_pred, average=None)
    recall_class = recall_score(Y_test, Y_pred, average=None)

    print("Acc: " + str(accuracy_class))
    print("f1_score: " + str(f1_class))
    print("precision: " + str(precision_class))
    print("recall: " + str(recall_class))

    #if not use_existing_split:
        #plot_history(history)

    print("")

    # Return train and test accuracy
    #return history.history['acc'][-1], history.history['val_acc'][-1], accuracy
    return accuracy, accuracy_class, f1_class, precision_class, recall_class, model, t1-t0

def cnn_predict_grid(data_in=None, 
            win_sizes=[((int(8), int(5)), 2, 1),((int(10), int(6)), 3, 2),((int(13), int(8)), 4, 3)], 
            problim = 0.95,
            model_fpath=model_fpath,
            scaler_fpath=scaler_fpath,
            nc_fpath='D:/Master/data/cmems_data/global_10km/noland/phys_noland_2016_060.nc',
            storedir=None):

    """ Test the model using multiple sliding windows, there will be multiple returned predictions
    data in: [lon,lat,x,y,ssl,uvel,vvel]
    storedir: path to directory for storing image of predicted grid, if None, no image is stored"""

    print("\n\n")

    lon,lat,x,y,ssl,uvel,vvel = data_in

    # Recreate the exact same model purely from the file

    custom_objects  = {
        "f1_m": f1_m,
        "precision_m": precision_m, 
        "recall_m": recall_m
    }

    clf = load_model(model_fpath, custom_objects=custom_objects)
    scaler = joblib.load(scaler_fpath) # Import the std sklearn scaler model
    
    nx, ny = ssl.shape 

    # Create canvas to show the cv2 rectangles around predictions
    fig, ax = plt.subplots(figsize=(15, 12))
    n=-1
    color_array = np.sqrt(((uvel.T-n)/2)**2 + ((vvel.T-n)/2)**2)
    # x and y needs to be equally spaced for streamplot
    if not (same_dist_elems(x) or same_dist_elems(y)):
        x, y = np.arange(len(x)),  np.arange(len(y)) 
    ax.contourf(x, y, ssl.T, cmap='rainbow', levels=150)
    ax.streamplot(x, y, uvel.T, vvel.T, color=color_array, density=10) 
    #ax.quiver(x, y, uvel.T, vvel.T, scale=3) 
    fig.subplots_adjust(0,0,1,1)
    fig.canvas.draw()

    im = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    im = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    imCopy = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
    imH, imW, _ = imCopy.shape # col, row
    winScaleW, winScaleH = imW*1.0/nx, imH*1.0/ny # Scalar coeff from dataset to cv2 image

    # Define what variables are used as channel, if only uvel and vvel it should be [1,2]
    to_be_scaled = [1,2] 
    data = [ssl, uvel, vvel]

    # Holds rectangle coordinates with dataset and image indexes
    cyc_r, acyc_r = [], []
    cyc_r_im, acyc_r_im = [], []

    print("++ Performing sliding window and predicting using pre-trained CNN model")
    # Loop over different window sizes, they will be resized down to correct dimensiona anyways
    for wSize, wStep, hStep in win_sizes:
        # loop over the sliding window of indeces
        for rectIdx, (i, j, (xIdxs, yIdxs)) in enumerate(sliding_window(ssl, wStep, hStep, windowSize=wSize)):

            if xIdxs[-1] >= nx or yIdxs[-1] >= ny:
                continue

            winW2, winH2 = winW*6, winH*6
            winSize = (winH2, winW2)

            masked = False # Continue if window hits land

            data_window, data_scaled_window = [], []

            for c in range(len(data)):
                # Creates window, checks if masked, if not returns the window
                a = check_window(data[c], xIdxs, yIdxs)
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
                    k = len(data_scaled_window) - 1
                    # Flatten array before applying scalar
                    data_scaled_window[k] = data_scaled_window[k].flatten()
                    # Scale the data
                    data_scaled_window[k] = scaler[k].transform([data_scaled_window[k]])[0]
                    # Reshape scaled data to original shape
                    data_scaled_window[k] = data_scaled_window[k].reshape(winW2, winH2)
            
            # continue to next window if mask (land) is present
            if masked: continue

            # Transfrom input window to CNN input format 
            X_cnn = np.zeros((1,winW2,winH2,nChannels))
            for lo in range(winW2): # Row
                for la in range(winH2): # Column
                    for c in range(nChannels): # Channels
                        X_cnn[0,lo,la,c] = data_scaled_window[c][lo,la]

            # Predict and receive probability
            prob = clf.predict(X_cnn)

            # This is the size of the current sliding window
            nxWin, nyWin = len(xIdxs), len(yIdxs)

            # y starts in top left for cv2, want it to be bottom left
            xr, yr = int(winScaleW*(i)), int(winScaleH*(ny-j)) # rect coords
            xrW, yrW= int(winScaleW*nxWin), int(winScaleH*nyWin) # rect width

            # If either cyclone or acyclone are above probability limit, we have a prediction
            if any(p >= problim for p in prob[0,1:]):       
                if prob[0,1] >= problim:
                    acyc_r.append([i, j, i + nxWin, j + nyWin])
                    acyc_r_im.append([xr, yr, xr + xrW, yr - xrW])
                    cv2.rectangle(imCopy, (xr, yr), (xr + xrW, yr - xrW), (217, 83, 25), 2)
                    #print('anti-cyclone | prob: {}'.format(prob[0,1]*100))
                else:
                    cyc_r.append([i, j, i + nxWin, j + nyWin])
                    cyc_r_im.append([xr, yr, xr + xrW, yr - xrW])
                    cv2.rectangle(imCopy, (xr, yr), (xr + xrW, yr - xrW), (0, 76, 217), 2)
                    #print('cyclone | prob: {}'.format(prob[0,2]*100))
                    
    # We  want to return both grouped and ungrouped predictions, in case user wants different grouping
    # Predictions need at least 2 rectangles with 20% overlap to be a final prediciton
    cyc_r_im_grouped, _ = cv2.groupRectangles(rectList=cyc_r_im, groupThreshold=1, eps=0.2) 
    acyc_r_im_grouped, _ = cv2.groupRectangles(rectList=acyc_r_im, groupThreshold=1, eps=0.2)

    # if a store directory is defined, create and store an image of both grouped and ungrouped 
    # predicted grid at location
    imgdir = 'C:/Users/47415/Master/images/compare/'
    if isinstance(storedir, str):
        if not os.path.isdir(imgdir + storedir):
            os.makedirs(imgdir + storedir)

        cv2.imwrite(imgdir + f'{storedir}/full_pred_grid.png', imCopy)
        imCopy = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        
        draw_rectangles(imCopy, cyc_r_im_grouped, lon, lat, winScaleW, winScaleH, 'cyclone')
        draw_rectangles(imCopy, acyc_r_im_grouped, lon, lat, winScaleW, winScaleH, 'anti-cyclone')

        cv2.imwrite(imgdir + f'{storedir}/grouped_pred_grid.png', imCopy)
        #cv2.imshow("Window", imCopy)
        #cv2.waitKey(0)

    plt.close(fig)
    return cyc_r, acyc_r, cyc_r_im_grouped, acyc_r_im_grouped


def real_time_test(problim=0.95):
    """ Only used for testing real-time predictions, super slow on large grids if multiple windows used."""

    latitude = [45.9, 49.1]
    longitude = [-23.2, -16.5]
    #download.download_nc(longitude, latitude)

    fig, ax = plt.subplots(figsize=(12, 8))     
    fig.subplots_adjust(0,0,1,1)
    
    clf = load_model(model_fpath)

    ncdir = "D:/Master/data/cmems_data/global_10km/noland/realtime/"
    for imId, fName in enumerate(os.listdir(ncdir)):

        lon,lat,sst,ssl,uvel,vvel =  load_nc_phys(ncdir + fName)
        nx, ny = ssl.shape 

        ax.clear()
        ax.contourf(lon, lat, ssl.T, cmap='rainbow', levels=100)
        ax.contour(lon, lat, ssl.T,levels=50)#,colors='k')#,linewidth=0.001)
        n=-1
        color_array = np.sqrt(((uvel-n)/2)**2 + ((vvel-n)/2)**2)
        ax.quiver(lon, lat, uvel.T, vvel.T, color_array)#, scale=12) #units="xy", ) # Plot vector field 
        fig.canvas.draw()

        im = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        im = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        imCopy = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        imH, imW, _ = imCopy.shape # col, row
        winScaleW, winScaleH = imW/nx, imH/ny # Scalar coeff from dataset to cv2 image

        to_be_scaled = [0,1,2] # Only use uvel and vvel
        data = [ssl, uvel, vvel]
        scaler = joblib.load(scaler_fpath) # Import the std sklearn scaler model
        cycloneRects, anticycloneRects = [], []

        # Loop over different window sizes, they will be resized down to correct dimensiona anyways
        #for wSize, wStep, hStep in [((int(7), int(4)), 2, 1), 
                            #((int(10), int(6)), 3, 2), 
                            #((int(12), int(7)), 4, 2)]:

        for wSize, wStep, hStep in [((int(11), int(6)), 3, 2),]:      

            # loop over the sliding window of indeces
            for rectIdx, (x, y, (lonIdxs, latIdxs)) in enumerate(sliding_window(ssl, wStep, hStep, windowSize=wSize)):

                if lonIdxs[-1] >= nx or latIdxs[-1] >= ny:
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
                        k = len(data_scaled_window) - 1
                        # Flatten array before applying scalar
                        data_scaled_window[k] = data_scaled_window[k].flatten()
                        # Scale the data
                        data_scaled_window[k] = scaler[k].transform([data_scaled_window[k]])[0]
                        # Reshape scaled data to original shape
                        data_scaled_window[k] = data_scaled_window[k].reshape(winW2, winH2)
                
                # continue to next window if mask (land) is present
                if masked: continue

                x_, y_ = int(winScaleW*(x)), int(winScaleH*(ny-y)) # y starts in top left for cv2, want it to be bottom left
                winW_, winH_= int(winScaleW*winW), int(winScaleH*winH)

                X_cnn = np.zeros((1,winW2,winH2,nChannels))
                for lo in range(winW2): # Row
                    for la in range(winH2): # Column
                        for c in range(nChannels): # Channels
                            X_cnn[0,lo,la,c] = data_scaled_window[c][lo,la]

                # Predict and receive probability
                prob = clf.predict(X_cnn)

                if any(p >= problim for p in prob[0,1:]):
                    if prob[0,2] >= problim:
                        anticycloneRects.append([x_, y_, x_ + winW_, y_ - winH_])
                        cv2.rectangle(imCopy, (x_, y_), (x_ + winW_, y_ - winH_), (217, 83, 25), 2)
                        print('anti-cyclone | prob: {}'.format(prob[0,1]*100))
                    else:
                        cycloneRects.append([x_, y_, x_ + winW_, y_ - winH_])
                        cv2.rectangle(imCopy, (x_, y_), (x_ + winW_, y_ - winH_), (0, 76, 217), 2)
                        print('cyclone | prob: {}'.format(prob[0,2]*100))

        cycloneRects, _ = cv2.groupRectangles(rectList=cycloneRects, groupThreshold=1, eps=0.2)
        anticycloneRects, _ = cv2.groupRectangles(rectList=anticycloneRects, groupThreshold=1, eps=0.2)
        for r in anticycloneRects:

            cv2.rectangle(im, (r[0], r[1]), (r[2], r[3]), (217, 83, 25), 2)

            textLabel = 'anti-cylcone'
            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
            textOrg = (r[0], r[1])
            cv2.rectangle(im, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
            cv2.rectangle(im, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(im, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
        
        for r in cycloneRects:
            cv2.rectangle(im, (r[0], r[1]), (r[2], r[3]), (0, 76, 217), 2)
            textLabel = 'cylcone'
            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
            textOrg = (r[0], r[1])
            cv2.rectangle(im, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
            cv2.rectangle(im, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(im, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
        
        cv2.imshow("Window", im)
        cv2.waitKey(1)
        cv2.imwrite(f'D:/master/TTK-4900-Master/images/realtime/cnn_pred{imId}.png', im)
        print("\n\n")

    plt.clf()


def plot_feature_map():
    """ Plotting feature maps created by the CNN model, only for report testing """

    data = []

    dirpath = 'C:/Users/47415/Master/TTK-4900-Master/data/'
    with np.load(dirpath+'uvel_train.npz', allow_pickle=True) as h5f:
        data.append(h5f['arr_0'][293,0])
    with np.load(dirpath+'vvel_train.npz', allow_pickle=True) as h5f:
        data.append(h5f['arr_0'][293,0])

    #fig, ax = plt.subplots(1,2,figsize=(10,8))
    #ax[0].quiver(uvel.T, vvel.T, scale=0.9) 

    winW2, winH2 = winW*6, winH*6
    cnn_data = np.zeros((1, winW2, winH2, 2))

    custom_objects  = {
        "f1_m": f1_m,
        "precision_m": precision_m, 
        "recall_m": recall_m
    }

    model_fpath = 'C:/Users/47415/Master/TTK-4900-Master/models/best_model_975.h5'
    clf = load_model(model_fpath, custom_objects=custom_objects)

    scaler = joblib.load(scaler_fpath) # Import the std sklearn scaler model

    for i in range(2):
        tmp = cv2.resize(data[i], dsize=(winH2, winW2), interpolation=cv2.INTER_CUBIC)
        tmp = tmp.flatten()
        tmp = scaler[i].transform([tmp])[0]
        cnn_data[0,:,:,i] = tmp.reshape(winW2, winH2)

    from keras import models
    # Extracts the outputs of the top 12 layers
    layer_outputs = [layer.output for layer in clf.layers[:12]] 
    # Creates a model that will return these outputs, given the model input
    activation_model = models.Model(inputs=clf.input, outputs=layer_outputs) 

    activations = activation_model.predict(cnn_data) 

    first_layer_activation = activations[0]
    print(first_layer_activation.shape)
    # Try plotting the fourth channel
    plt.matshow(first_layer_activation[0, :, :, 0], cmap='viridis')

    plt.show()


if __name__ == '__main__':
    """ If models are to be run directly """
    #train_model(use_existing_split=False) 
    #find_best_model()
    #find_average_model()
    #analyse_h5()  
    #test_model()
    #real_time_test()
    #plot_feature_map()