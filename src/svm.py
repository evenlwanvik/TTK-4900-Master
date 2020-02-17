from tools.sliding_window import localize_and_classify
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from tools.load_nc import load_netcdf4
#from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from tools import gui, dim
from tools.machine_learning import getAccuracy, preprocess_data
import numpy as np
import pickle

data_path = 'C:/Master/TTK-4900-Master/data/training_data/200_days_2018/ssl_train.npz'

model_fpath = 'models/svm_ssl_01.sav'


def train_model(data_path=data_path, model_fpath=model_fpath):
    
    # Get the training data
    X_train, X_test, y_train, y_test = preprocess_data(data_path)

    shape = X_train.shape
    X_train = X_train.reshape(shape[0],shape[1]*shape[2])
    shape = X_test.shape
    X_test = X_test.reshape(shape[0],shape[1]*shape[2])
    

    '''
    # HOG?
    feature_descriptor, img_hog = hog(ssl_x[5], orientations=5, pixels_per_cell=(2,2), cells_per_block=(1,1), visualize=True)
    img_hog = np.divide(img_hog, 255.0)
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axs[0,0].imshow(ssl_x[5], cmap='gray')
    axs[0,1].imshow(phase_x[5], cmap='CMRmap')
    #axs[1,0].quiver(None, None, data[2].T, data[3].T, color_array, scale=7)
    axs[1,1].imshow(img_hog, cmap='gray')
    guiEvent, guiValues = gui.show_figure(fig)
    plt.close(fig)
    '''
    pipeline = OneVsRestClassifier(SVC(kernel='poly', verbose=1, probability=True))

    parameters = {
            'estimator__gamma': [0.1, 0.5, 1, 10, 100],
            'estimator__C': [0.1, 1, 10, 100, 1000],
            'estimator__kernel': ['poly', 'rbf'],
            'estimator__degree': [0, 1, 2, 3, 4, 5, 6]
    }   

    # Create a classifier object with the classifier and parameter candidates
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=2, verbose=3, scoring="accuracy")

    model.fit(list(X_train), y_train)      

    pickle.dump(model, open(model_fpath, 'wb'))

    y_pred = model.predict(list(X_test))

    accuracy = getAccuracy(y_pred, y_test)
    print(f"> The accuracy of the model is {accuracy}")


def test_model(nc_fpath='C:/Master/data/cmems_data/global_10km/phys_noland_001.nc', model_fpath=model_fpath):

    (ds,t,lon,lat,depth,uvel_full,vvel_full,sst_full,ssl_full) =  load_netcdf4(nc_fpath)
    sst = np.array(sst_full[0].T, dtype='float32') # NC uses (lat,lon), we want (lon/lat) and it needs to be float32
    loaded_model = pickle.load(open(model_fpath, 'rb'))
    
    localize_and_classify(sst, loaded_model, 'svm') 


if __name__ == '__main__':
    #test_model()
    train_model()
