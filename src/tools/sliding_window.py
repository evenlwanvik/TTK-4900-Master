import time
import cv2
from PIL import Image
import numpy as np

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
    for y in range(0, dims[0], stepSize):
       for x in range(0, dims[1], stepSize): 
           # yield current window
           yield(x, y, arr[y : y+windowSize[1], x : x+windowSize[0]])


def localize_and_classify(data, clf, clf_type='svm', probLim=0.9, stepSize=5, winW=20, winH=20, draw_window=False):
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

    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(data, stepSize=stepSize, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        if clf_type == 'svm':
            probabilities = clf.predict_proba(list(window.flatten()/255.0))
        elif clf_type == 'cnn':

            #window = cv2.resize(X[i], dsize=(nLat, nLon), interpolation=cv2.INTER_CUBIC)/255.0
            window_cnnshape = window.reshape(1, 20, 20, 1)/255.0

            probabilities = clf.predict(window_cnnshape) 


        for i, prob in enumerate(probabilities[0]):
            if prob > probLim:
                print(prob)
                predictions.append((window, helpers.numToChar(i), prob))
                color = (0, 255, 0)
                cv2.rectangle(display_data, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            else:
                color = (0, 0, 255)




        # Draw the window
        if draw_window:
            clone = cv2.cvtColor(data.copy(), cv2.COLOR_GRAY2RGB)
            cv2.rectangle(clone*255.0, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            time.sleep(0.025)

    for i, pred in enumerate(predictions):
        if pred == 1:         
            print("cyclone ("+pred+") at idx "+i)
        elif pred == -1:
            print("anti-cyclone ("+pred+") at idx "+i)
    
    cv2.imshow("Prediction", display_data)
    cv2.waitKey(0)

    #return display_image