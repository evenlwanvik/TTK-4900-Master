import time
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def sliding_window(arr, stepSize, windowSize):
    ''' 
    slide window across the image 
    args:
        arr         (float): 2d (image) array
        stepSize    (int): number of pixels to skip per new window
        windowSize  (int): width and height of window in pixels
    returns:
        generator of windows
    ''' 
    dims = arr.shape

    for x in range(0, dims[0], stepSize):
       for y in range(0, dims[1], stepSize): 
           # yield current window
           yield x, y, (list(range(x, x+windowSize[0])), list(range(y, y+windowSize[1])))

def localize_and_classify(data, lon, lat, clf, clf_type='svm', probLim=0.6, stepSize=5, winW=20, winH=20, draw_window=False):
    ''' 
    Print letter classifications and plot their location
    args:
        data        (2d array): 2d array in
        clf         (classifier): trained ML classifier
        clf_type    (string): 'svm' or 'cnn'
        probLim     (float): model's confidence needs to surpass this limit
        stepSize    (int): stepsize of window, e.g. slide window+stepsize
        winW        (int): width of image
        winH        (int): height of image
        draw_window (bool): if you want to see the sliding window
    returns:
        image with windows with supposedly letters in them 
    '''


    #print(f"> Initiating sliding window with for {} with:\n confidence threshold: {probLim*100}%\n step size: {stepSize}\n window dimension: {winW}x{winH}").

    predictions = []
    display_data = cv2.cvtColor(data.copy(), cv2.COLOR_GRAY2RGB)
    color = (0, 0, 255) # use red as window default, only turn green when detect letter
    shape = data[0].shape

    scaler = MinMaxScaler(feature_range=(0,1))
    data = scaler.fit_transform(data)

    # loop over the sliding window for each layer of the pyramid
    for (x, y, (lonIdxs, latIdxs)) in sliding_window(data, stepSize=stepSize, windowSize=(winW, winH)):

        #print(latIdxs[-1])
        if lonIdxs[-1] > shape[0] or latIdxs[-1] > shape[1]:
            continue
        
        #print(f"lon shape: {np.array(lonIdxs).shape}, lat shape: {np.array(latIdxs).shape}, x:{x}, y:{y}")
        window = np.array([[data[i,j] for j in latIdxs] for i in lonIdxs])

        lo, la = lon[lonIdxs], lat[latIdxs]
        # if the window does not meet our desired window size, ignore it


        if clf_type == 'svm':
            # probabilities = clf.predict_proba(np.expand_dims(np.array(window.flatten()), 0))
            probabilities = clf.predict_proba( [np.array(window.flatten())] )
            #print(probabilities)



            print(probabilities)
            #probabilities = clf.predict_proba([np.array(window.flatten())])
            if probabilities[0,1] > probabilities[0,0] and probabilities[0,1] > probLim:
                print(probabilities)
                fig, ax = plt.subplots(figsize=(14, 10), dpi= 80, facecolor='w', edgecolor='w')
                plt.contourf(lo, la, window.T, 20)
                plt.show()           

    '''    
        
        elif clf_type == 'cnn':

            #window = cv2.resize(X[i], dsize=(nLat, nLon), interpolation=cv2.INTER_CUBIC)/255.0
            window_cnnshape = window.reshape(1, 20, 20, 1)

            probabilities = clf.predict(window_cnnshape) 


        print(probabilities)
        for i, prob in enumerate(probabilities[0]):
            if prob > probLim:
                predictions.append((window, i, prob))
                color = (0, 255, 0)
                cv2.rectangle(display_data, (x, y), (x + winW, y + winH), color, 2)
            else:
                color = (0, 0, 255)
                
        # Draw the window
        if draw_window:
            clone = cv2.cvtColor(data.copy(), cv2.COLOR_GRAY2RGB)
            cv2.rectangle(clone*255.0, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            time.sleep(0.1)
    exit()     
    for i, pred in enumerate(predictions):

        if pred[1] == 0:         
            print(f"non-eddie ({pred[1]}) at idx {i}")
        elif pred[1] == 1:
            print(f"Eddie ({pred[1]}) at idx {i}")
    
    cv2.imshow("Prediction", display_data)
    cv2.waitKey(0)

    #return display_image
    '''   