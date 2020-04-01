from tools.machine_learning import getAccuracy, preprocess_data, sliding_window
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from tools.load_nc import load_nc_data
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import pickle
import cv2
import os

from sklearn.externals import joblib # To save scaler



sst_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/sst_train.npz'
ssl_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/ssl_train.npz'
uvel_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/uvel_train.npz'
vvel_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/vvel_train.npz'
phase_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/phase_train.npz'
lon_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/lon.npz'
lat_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/lat.npz'
model_fpath = 'D:/master/models/2016/svm_mult_full.h5'
scaler_fpath = "D:/master/models/2016/svm_norm_scaler.pkl"
#2016/new
#200_days_2018

# Create a scaler for each channel
nChannels = 2
scaler = [StandardScaler() for _ in range(nChannels)]
#scaler = MinMaxScaler(feature_range=(-1,1))
winW, winH = int(11), int(6)
probLim = 0.93

def train_model():

    winW2, winH2 = winW*4, winH*4

    # TODO: This part is the same for every model if we use 3 channels, create function
    # Open the numpy training data array and append for each channel we want to use
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

    nTeddies = len(X[0])

    for c in range(nChannels): # For each channel
        for i in range(nTeddies): # For each Training Eddy
            #amin, amax = np.amin(X[c][i]), np.amax(X[c][i])
            #X[c][i] = X[c][i]/90
            X[c][i] = cv2.resize(X[c][i], dsize=(winH2, winW2), interpolation=cv2.INTER_CUBIC) 

    # Reshape data for CNN (sample, width, height, channel)
    X_svm = np.zeros((nTeddies,winW2,winH2,nChannels))
    for i in range(nTeddies): # Eddies
        for lo in range(winW2): # Row
            for la in range(winH2): # Column
                for c in range(nChannels): # Channels
                    X_svm[i,lo,la,c] = X[c][i][lo][la]

    # Create and set the scaler for each channel
    X_svm = X_svm.reshape(nTeddies, -1, nChannels)
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
            'estimator__gamma': [0.00005, 0.0001, 0.0005, 0.001],
            'estimator__C': [1, 10, 100],
            'estimator__kernel': ['rbf'],
            #'estimator__degree': [2, 3, 4, 5],
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


def test_model(nc_fpath='D:/Master/data/cmems_data/global_10km/2016/noland/phys_noland_2016_060.nc'):
    
    # Download grid to be tested
    latitude = [45.9, 49.1]
    longitude = [-23.2, -16.5]
    #latitude = [45, 50]
    #longitude = [-24, -12]
    #download.download_nc(longitude, latitude)

    # Test cv2 image and sliding window movement on smaller grid
    nc_fpath='D:/Master/data/cmems_data/global_10km/2016/noland/smaller/phys_noland_2016_001.nc'
    
    print("\n\n")

    lon,lat,sst,ssl,uvel,vvel =  load_nc_data(nc_fpath)

    day = 0

    # Recreate the exact same model purely from the file
    model = pickle.load(open(model_fpath, 'rb'))

    nLon, nLat = ssl.shape 
    wSize = (winW, winH)
    wStep, hStep = 6, 3 

    print("\n\nperforming sliding window on satellite data \n\n")

    # Create canvas to show the cv2 rectangles around predictions
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    # Canvas for zooming in on prediction
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    fig2.show()

    # Draw on the larger canvas before sliding
    ax1.contourf(lon, lat, ssl.T, cmap='rainbow', levels=60)
    ax1.contour(lon, lat, ssl.T,levels=60)#,colors='k')#,linewidth=0.001)
    n=-1
    color_array = np.sqrt(((uvel-n)/2)**2 + ((vvel-n)/2)**2)
    ax1.quiver(lon, lat, uvel.T, vvel.T, color_array)#, scale=12) #units="xy", ) # Plot vector field      
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
    for rectIdx, (x, y, (lonIdxs, latIdxs)) in enumerate(sliding_window(ssl, wStep, hStep, windowSize=(wSize))):

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

        X_svm = np.zeros((1,winW2,winH2,nChannels))
        for lo in range(winW2): # Row
            for la in range(winH2): # Column
                for c in range(nChannels): # Channels
                    X_svm[0,lo,la,c] = data_scaled_window[c][lo,la]

        # Flatten array
        X_svm = X_svm.reshape(1,-1)

        # Predict and receive probability
        prob = model.predict_proba(X_svm)

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
    


def real_time_test():

    latitude = [45.9, 49.1]
    longitude = [-23.2, -16.5]
    #download.download_nc(longitude, latitude)

    fig1, ax1 = plt.subplots(figsize=(12, 8))     
    fig1.subplots_adjust(0,0,1,1)
    
    model = pickle.load(open(model_fpath, 'rb'))

    probLim = 0.92

    ncdir = "D:/Master/data/cmems_data/global_10km/2016/noland/realtime/"
    for imId, fName in enumerate(os.listdir(ncdir)):

        lon,lat,sst,ssl,uvel,vvel =  load_nc_data(ncdir + fName)
        nLon, nLat = ssl.shape 

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
        winScaleW, winScaleH = imW/nLon, imH/nLat # Scalar coeff from dataset to cv2 image

        to_be_scaled = [1,2] # Only use uvel and vvel
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

                X_svm = np.zeros((1,winW2,winH2,nChannels))
                for lo in range(winW2): # Row
                    for la in range(winH2): # Column
                        for c in range(nChannels): # Channels
                            X_svm[0,lo,la,c] = data_scaled_window[c][lo,la]

                # Flatten array
                X_svm = X_svm.reshape(1,-1)
                # Predict and receive probability
                prob = model.predict_proba(X_svm)

                if any(i >= probLim for i in prob[0,1:]):
                    if prob[0,1] >= probLim:
                        anticycloneRects.append([x_, y_, x_ + winW_, y_ - winH_])
                        cv2.rectangle(imCopy, (x_, y_), (x_ + winW_, y_ - winH_), (217, 83, 25), 2)
                        print('anti-cyclone | prob: {}'.format(prob[0,1]*100))
                    else:
                        cycloneRects.append([x_, y_, x_ + winW_, y_ - winH_])
                        cv2.rectangle(imCopy, (x_, y_), (x_ + winW_, y_ - winH_), (0, 76, 217), 2)
                        print('cyclone | prob: {}'.format(prob[0,2]*100))

        cycloneRects, _ = cv2.groupRectangles(rectList=cycloneRects, groupThreshold=1, eps=0.5)
        anticycloneRects, _ = cv2.groupRectangles(rectList=anticycloneRects, groupThreshold=1, eps=0.5)
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
        cv2.imwrite(f'D:/master/TTK-4900-Master/images/realtime/svm_pred{imId}.png', im)
        print("\n\n")




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
    real_time_test()