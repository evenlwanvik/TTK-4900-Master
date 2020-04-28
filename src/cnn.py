from tools.machine_learning import sliding_window
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from tools.load_nc import load_nc_cmems
from sklearn.externals import joblib # To save scaler
from keras.models import load_model
import matplotlib.pyplot as plt
from tools import dim
import xarray as xr
import numpy as np
from tools import cnn_models
import cv2
import time
import os



##################### TOOLS #####################

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

    import zipfile
    import scipy.io
    import h5py

    dirpath = 'C:/Users/47415/Master/TTK-4900-Master/data/training_data/2016/h5/'
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

def plot_OW():
    from tools.eddies import load_netcdf4,eddy_detection

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

##################### TRAIN AND TEST #####################


sst_path = 'C:/Users/47415/Master/TTK-4900-Master/data/training_data/2016/sst_train.npz'
ssl_path = 'C:/Users/47415/Master/TTK-4900-Master/data/training_data/2016/ssl_train.npz'
uvel_path = 'C:/Users/47415/Master/TTK-4900-Master/data/training_data/2016/uvel_train.npz'
vvel_path = 'C:/Users/47415/Master/TTK-4900-Master/data/training_data/2016/vvel_train.npz'
phase_path = 'C:/Users/47415/Master/TTK-4900-Master/data/training_data/2016/phase_train.npz'
lon_path = 'C:/Users/47415/Master/TTK-4900-Master/data/training_data/2016/lon.npz'
lat_path = 'C:/Users/47415/Master/TTK-4900-Master/data/training_data/2016/lat.npz'
model_fpath = 'D:/Master/models/2016/cnn_mult_full.h5'
scaler_fpath = "D:/Master/models/2016/cnn_norm_scaler.pkl"

# Create a scaler for each channel
nChannels = 2
scaler = [StandardScaler() for _ in range(nChannels)]
#scaler = MinMaxScaler(feature_range=(-1,1))
winW, winH = int(11), int(6)

# Fortsett å endre oppløsning for å se om vi kan ha mindre, *4 var best
# Også prøv forskjellige kombinasjoner av kanaler


def train_model():

    winW2, winH2 = winW*4, winH*4

    X = np.array([])

    # Open the numpy training data array and append for each channel we want to use
    X = []
    #with np.load(sst_path, allow_pickle=True) as data:
    #    X.append(data['arr_0'][:,0])    
    #with np.load(ssl_path, allow_pickle=True) as data:
    #    X.append(data['arr_0'][:,0])
    #    Y = data['arr_0'][:,1]
    with np.load(uvel_path, allow_pickle=True) as data:
        X.append(data['arr_0'][:,0])
        Y = data['arr_0'][:,1]
    with np.load(vvel_path, allow_pickle=True) as data:
        X.append(data['arr_0'][:,0])
    #with np.load(vvel_path, allow_pickle=True) as data:
    #    X.append(data['arr_0'][:,0])
    #    Y = data['arr_0'][:,1]
    #with np.load(phase_path, allow_pickle=True) as data:
    #    X.append(data['arr_0'][:,0]) 
    #    Y = data['arr_0'][:,1]       


    # Reshape the "image" to standard size
    nTeddies = len(X[0])
    
    #X[np.isnan(X)] = 0

    for c in range(nChannels): # For each channel
        for i in range(nTeddies): # For each Training Eddy
            X[c][i] = np.array(X[c][i], dtype='float32')
            X[c][i][np.isnan(X[c][i])] = 0 # If we have land present, just set to zero TODO: change it?
            X[c][i] = cv2.resize(X[c][i], dsize=(winH2, winW2), interpolation=cv2.INTER_CUBIC) 

    # Reshape data (sample, width, height, channel) 
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

    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X_cnn, Y[:nTeddies], test_size=0.33)
    nTrain = len(X_train)
    
    input_shape = (winW2, winH2, nChannels)
    #model = cnn_models.mnist(input_shape=input_shape, classes=3)
    #model = cnn_models.VGG16(input_shape=input_shape, classes=3)
    #model = cnn_models.my_model(input_shape = input_shape, classes = 3)
    #model = cnn_models.my_model_inception(input_shape = input_shape, classes = 3)
    #model, callbacks_list = cnn_models.inception_resnet_v2(input_shape=input_shape,classes=3,model_fpath=model_fpath)
    model = cnn_models.best_sofar(input_shape=input_shape,classes=3)
    #model = cnn_models.best_sofar_resnet(input_shape=input_shape,classes=3)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adagrad', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    
    # Create 3 columns for each class for multilabel classification
    Y_train = convert_to_one_hot(Y_train)
    Y_test  = convert_to_one_hot(Y_test)

    print('\n\n')
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    print('\n\n')

    #history = model.fit(X_train, Y_train, validation_split=0.33, epochs = 25, batch_size = 1, callbacks=callbacks_list)
    history = model.fit(X_train, Y_train, validation_split=0.33, epochs = 40, batch_size = 1)
    
    preds = model.evaluate(X_test, Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
    print('\n\n')

    y_pred = model.predict(X_test)
    acc = np.equal(np.argmax(Y_test, axis=-1), np.argmax(y_pred, axis=-1)).mean()
    print(acc)

    model.save(model_fpath)

    plot_history(history)

from cmems_download import download

def cnn_predict_grid(data_in=None, 
            win_sizes=[((int(8), int(5)), 2, 1),((int(10), int(6)), 3, 2),((int(13), int(8)), 4, 3)], 
            problim = 0.95,
            model_fpath=model_fpath,
            nc_fpath='D:/Master/data/cmems_data/global_10km/2016/noland/phys_noland_2016_060.nc',
            storedir=None):

    print("\n\n")

    lon,lat,x,y,ssl,uvel,vvel = data_in

    # Recreate the exact same model purely from the file
    clf = load_model(model_fpath)
    #ssl_clf   = keras.models.load_model(D:/master/models/2016/cnn_{}class_ssl.h5'.format(cnntype))

    nx, ny = ssl.shape 

    # Create canvas to show the cv2 rectangles around predictions
    fig1, ax1 = plt.subplots(figsize=(15, 12))
    n=-1
    color_array = np.sqrt(((uvel.T-n)/2)**2 + ((vvel.T-n)/2)**2)
    # x and y needs to be equally spaced for streamplot
    if not (same_dist_elems(x) or same_dist_elems(y)):
        x, y = np.arange(len(x)),  np.arange(len(y)) 
    ax1.contourf(x, y, ssl.T, cmap='rainbow', levels=150)
    ax1.streamplot(x, y, uvel.T, vvel.T, color=color_array, density=10) 
    #ax1.quiver(x, y, uvel.T, vvel.T, scale=3) 
    fig1.subplots_adjust(0,0,1,1)
    fig1.canvas.draw()

    im = np.frombuffer(fig1.canvas.tostring_rgb(), dtype=np.uint8)
    im = im.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
    imCopy = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
    imH, imW, _ = imCopy.shape # col, row
    winScaleW, winScaleH = imW*1.0/nx, imH*1.0/ny # Scalar coeff from dataset to cv2 image

    # Only use uvel and vvel to be scaled and use for CNN
    to_be_scaled = [1,2] 
    data = [ssl, uvel, vvel]

    scaler = joblib.load(scaler_fpath) # Import the std sklearn scaler model

    # Holds rectangle coordinates with dataset and image indexes
    cyc_r, acyc_r = [], []
    cyc_r_im, acyc_r_im = [], []

    print("\n\nperforming sliding window on satellite data \n\n")
    # Loop over different window sizes, they will be resized down to correct dimensiona anyways
    for wSize, wStep, hStep in win_sizes:
        # loop over the sliding window of indeces
        for rectIdx, (i, j, (xIdxs, yIdxs)) in enumerate(sliding_window(ssl, wStep, hStep, windowSize=wSize)):

            if xIdxs[-1] >= nx or yIdxs[-1] >= ny:
                continue

            winW2, winH2 = winW*4, winH*4
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

            X_cnn = np.zeros((1,winW2,winH2,nChannels))
            for lo in range(winW2): # Row
                for la in range(winH2): # Column
                    for c in range(nChannels): # Channels
                        X_cnn[0,lo,la,c] = data_scaled_window[c][lo,la]

            # Predict and receive probability
            prob = clf.predict(X_cnn)

            # y starts in top left for cv2, want it to be bottom left
            xr, yr = int(winScaleW*(i)), int(winScaleH*(ny-j)) # rect coords
            xrW, yrW= int(winScaleW*winW), int(winScaleH*winH) # rect width

            if any(p >= problim for p in prob[0,1:]):       
                if prob[0,1] >= problim:
                    acyc_r.append([i, j, i + winW, j + winH])
                    acyc_r_im.append([xr, yr, xr + xrW, yr - xrW])
                    cv2.rectangle(imCopy, (xr, yr), (xr + xrW, yr - xrW), (217, 83, 25), 2)
                    #print('anti-cyclone | prob: {}'.format(prob[0,1]*100))
                else:
                    cyc_r.append([i, j, i + winW, j + winH])
                    cyc_r_im.append([xr, yr, xr + xrW, yr - xrW])
                    cv2.rectangle(imCopy, (xr, yr), (xr + xrW, yr - xrW), (0, 76, 217), 2)
                    #print('cyclone | prob: {}'.format(prob[0,2]*100))
                    
    # Group the rectangles according to how many and how much they overlap
    cyc_r_im_grouped, _ = cv2.groupRectangles(rectList=cyc_r_im, groupThreshold=1, eps=0.2)
    acyc_r_im_grouped, _ = cv2.groupRectangles(rectList=acyc_r_im, groupThreshold=1, eps=0.2)

    # if a store directory is defined, create and store image at location
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

    #cyc_r, _ = cv2.groupRectangles(rectList=cyc_r, groupThreshold=1, eps=0.2)
    #acyc_r, _ = cv2.groupRectangles(rectList=acyc_r, groupThreshold=1, eps=0.2)

    return cyc_r, acyc_r


def real_time_test(problim=0.95):

    latitude = [45.9, 49.1]
    longitude = [-23.2, -16.5]
    #download.download_nc(longitude, latitude)

    fig1, ax1 = plt.subplots(figsize=(12, 8))     
    fig1.subplots_adjust(0,0,1,1)
    
    clf = load_model(model_fpath)

    ncdir = "D:/Master/data/cmems_data/global_10km/2016/noland/realtime/"
    for imId, fName in enumerate(os.listdir(ncdir)):

        lon,lat,sst,ssl,uvel,vvel =  load_nc_cmems(ncdir + fName)
        nx, ny = ssl.shape 

        ax1.clear()
        ax1.contourf(lon, lat, ssl.T, cmap='rainbow', levels=100)
        ax1.contour(lon, lat, ssl.T,levels=50)#,colors='k')#,linewidth=0.001)
        n=-1
        color_array = np.sqrt(((uvel-n)/2)**2 + ((vvel-n)/2)**2)
        ax1.quiver(lon, lat, uvel.T, vvel.T, color_array)#, scale=12) #units="xy", ) # Plot vector field 
        fig1.canvas.draw()

        im = np.frombuffer(fig1.canvas.tostring_rgb(), dtype=np.uint8)
        im = im.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
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


if __name__ == '__main__':
    train_model() 
    #analyse_h5()  
    #test_model()
    #real_time_test()