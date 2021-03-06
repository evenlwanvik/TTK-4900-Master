from tools.machine_learning import getAccuracy, preprocess_data, sliding_window
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from tools.load_nc import load_nc_sat
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import pickle
import cv2
import os
from skimage.feature import hog

from sklearn.externals import joblib # To save scaler



sst_path = 'C:/Users/47415/Master/TTK-4900-Master/data/sst_train.npz'
ssl_path = 'C:/Users/47415/Master/TTK-4900-Master/data/ssl_train.npz'
uvel_path = 'C:/Users/47415/Master/TTK-4900-Master/data/uvel_train.npz'
vvel_path = 'C:/Users/47415/Master/TTK-4900-Master/data/vvel_train.npz'
phase_path = 'C:/Users/47415/Master/TTK-4900-Master/data/phase_train.npz'
lon_path = 'C:/Users/47415/Master/TTK-4900-Master/data/lon.npz'
lat_path = 'C:/Users/47415/Master/TTK-4900-Master/data/lat.npz'
model_fpath = 'C:/Users/47415/Master/TTK-4900-Master/models/svm_model.h5'
scaler_fpath = "C:/Users/47415/Master/TTK-4900-Master/models/svm_norm_scaler.pkl"
#new
#200_days_2018

# Create a scaler for each channel
nChannels = 2
scaler = [StandardScaler() for _ in range(nChannels)]
#scaler = MinMaxScaler(feature_range=(-1,1))
winW, winH = int(11), int(6)
probLim = 0.95

def train_model():

    winW2, winH2 = winW*4, winH*4

    X = []
    with np.load(uvel_path, allow_pickle=True) as data:
        X.append(data['arr_0'][:,0])
    with np.load(vvel_path, allow_pickle=True) as data:
        X.append(data['arr_0'][:,0])
        Y = data['arr_0'][:,1]

    nTeddies = len(X[0])

    for c in range(nChannels): # For each channel
        for i in range(nTeddies): # For each Training Eddy
            X[c][i] = cv2.resize(X[c][i], dsize=(winH2, winW2), interpolation=cv2.INTER_CUBIC) 

    # Reshape data for SVM (sample, width, height, channel)
    X_svm = np.zeros((nTeddies,winW2,winH2,nChannels))
    for i in range(nTeddies): # Eddies
        for lo in range(winW2): # Row
            for la in range(winH2): # Column
                for c in range(nChannels): # Channels
                    X_svm[i,lo,la,c] = X[c][i][lo][la]
   
    # Create and set the scaler for each channel
    #X_svm = X_svm.reshape(nTeddies, -1, nChannels)
    for c in range(nChannels):
        X_svm[:,:,c] = scaler[c].fit_transform(X_svm[:,:,c])
    joblib.dump(scaler, scaler_fpath) # Save the Scaler model

    # flatten each sample for svm, the method should be able to find the non-linear 
    # relationships between the seperate channels anyways.
    X_svm = X_svm.reshape(nTeddies, -1)
    
    for i in range(nTeddies):
        X_svm[i] = X_svm[i].flatten()

    # If land presetn (NaN), just set to zero
    X_svm = np.nan_to_num(X_svm)

    # Want classes to be from 0-2, I've used -1,0,1
    for i in range(len(Y)):
        if Y[i] == -1: Y[i]=2
    Y = Y.astype('int')

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_svm, Y[:nTeddies], test_size=0.33)

    pipeline = OneVsRestClassifier(SVC(kernel='rbf', verbose=1, probability=True))
    #pipeline = SVC(kernel='rbf', verbose=1, probability=True) # Single-class

    parameters = {
            #'estimator__gamma': [0.0001, 0.0003, 0.0006, 0.001],
            #'estimator__C': [1, 3, 6, 8],
            #'estimator__kernel': ['rbf'],
            'estimator__gamma': [0.01, 0.1, 1, 10],
            'estimator__C': [0.1, 1, 10],
            'estimator__kernel': ['rbf'],
    }       

    # Classifier object with the classifier and parameter candidates for cross-validated grid-search
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




def svm_predict_grid(data_in=None, 
            win_sizes=[((int(8), int(5)), 2, 1),((int(10), int(6)), 3, 2),((int(13), int(8)), 4, 3)], 
            problim = 0.95,
            model_fpath=model_fpath,
            nc_fpath='D:/Master/data/cmems_data/global_10km/noland/phys_noland_2016_060.nc',
            storedir=None):

    print("\n\n")

    lon,lat,x,y,ssl,uvel,vvel = data_in

    # Recreate the exact same model purely from the file
    model = pickle.load(open(model_fpath, 'rb'))
    #ssl_clf   = keras.models.load_model(C:/Users/47415/Master/TTK-4900-Master/models/cnn_{}class_ssl.h5'.format(cnntype))

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

    # Only use uvel and vvel to be scaled and use for CNN
    to_be_scaled = [1,2] 
    data = [ssl, uvel, vvel]

    scaler = joblib.load(scaler_fpath) # Import the std sklearn scaler model

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

            X_svm = np.zeros((1,winW2,winH2,nChannels))
            for lo in range(winW2): # Row
                for la in range(winH2): # Column
                    for c in range(nChannels): # Channels
                        X_svm[0,lo,la,c] = data_scaled_window[c][lo,la]

            # Flatten array
            X_svm = X_svm.reshape(1,-1)

            # Predict and receive probability
            prob = model.predict(X_svm)

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

    plt.close(fig)
    return cyc_r, acyc_r


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


if __name__ == '__main__':
    #train_model()
    #test_model()
    #real_time_test()