from tools.machine_learning import getAccuracy, preprocess_data, sliding_window
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from tools.load_nc import load_nc_sat
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
import h5py

from sklearn.externals import joblib # To save scaler


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



sst_path = 'C:/Users/47415/Master/TTK-4900-Master/data/sst_train.npz'
ssl_path = 'C:/Users/47415/Master/TTK-4900-Master/data/ssl_train.npz'
uvel_path = 'C:/Users/47415/Master/TTK-4900-Master/data/uvel_train.npz'
vvel_path = 'C:/Users/47415/Master/TTK-4900-Master/data/vvel_train.npz'
phase_path = 'C:/Users/47415/Master/TTK-4900-Master/data/phase_train.npz'
lon_path = 'C:/Users/47415/Master/TTK-4900-Master/data/lon.npz'
lat_path = 'C:/Users/47415/Master/TTK-4900-Master/data/lat.npz'
model_fpath = 'C:/Users/47415/Master/TTK-4900-Master/models/rf_model.h5'
scaler_fpath = "C:/Users/47415/Master/TTK-4900-Master/models/rf_norm_scaler.pkl"
#new
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

    # Create the parameter grid based on the results of random search 
    param_grid = {
        'n_estimators': [50, 100, 500, 1000],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Create a based model
    rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

    # Instantiate the grid search model
    model = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)

    
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

    print(model.best_params_)

def rf_predict_grid(data_in=None, 
            win_sizes=[((int(8), int(5)), 2, 1),((int(10), int(6)), 3, 2),((int(13), int(8)), 4, 3)], 
            problim = 0.95,
            model_fpath=model_fpath,
            nc_fpath='D:/Master/data/compare/phys_200.nc',
            storedir=None):

    print("\n\n")

    lon,lat,x,y,ssl,uvel,vvel = data_in

    # Recreate the exact same model purely from the file
    model = pickle.load(open(model_fpath, 'rb'))
  
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

            X_rf = np.zeros((1,winW2,winH2,nChannels))
            for lo in range(winW2): # Row
                for la in range(winH2): # Column
                    for c in range(nChannels): # Channels
                        X_rf[0,lo,la,c] = data_scaled_window[c][lo,la]

            # Flatten array
            X_rf = X_rf.reshape(1,-1)

            # Predict and receive probability
            prob = model.predict(X_rf)

            # y starts in top left for cv2, want it to be bottom left
            xr, yr = int(winScaleW*(i)), int(winScaleH*(ny-j)) # rect coords
            xrW, yrW= int(winScaleW*winW), int(winScaleH*winH) # rect width

            print(prob)
    
            if prob[0] == 1:
                acyc_r.append([i, j, i + winW, j + winH])
                acyc_r_im.append([xr, yr, xr + xrW, yr - xrW])
                cv2.rectangle(imCopy, (xr, yr), (xr + xrW, yr - xrW), (217, 83, 25), 2)
                #print('anti-cyclone | prob: {}'.format(prob[0,1]*100))
            if prob[0] == 2:
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

    


if __name__ == '__main__':
    train_model()
    #test_model()
    #real_time_test()