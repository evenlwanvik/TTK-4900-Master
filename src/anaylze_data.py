from tools import dim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from tools import gui
import cv2

def plot_training_data():
    ''' Just to analyze some of the npz training data before generating more training data '''
    dirpath = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/'

    sst = []
    ssl = []
    uvel = []
    vvel = []
    phase = []

    label = []

    
    with np.load(dirpath+'ssl_train.npz', allow_pickle=True) as h5f:
        # The standard grid size we will use
        gridSize = dim.find_avg_dim(h5f['arr_0'][:,0])
        nLon = gridSize[0]
        nLat = gridSize[1]

    nTeddies = 10 # number of training eddies to analyze
    for i in range(nTeddies):
        data = []
        for j, fpath in enumerate((dirpath+'sst_train.npz', dirpath+'ssl_train.npz', dirpath+'uvel_train.npz', dirpath+'vvel_train.npz', dirpath+'phase_train.npz')):
            with np.load(fpath, allow_pickle=True) as h5f:
                data.append(h5f['arr_0'][i,0])
                if j==0:
                    label = h5f['arr_0'][i,1]

            data[j] = np.array(data[j], dtype='float32') # convert to numpy array
            data[j] = cv2.resize(data[j], dsize=(nLat, nLon), interpolation=cv2.INTER_CUBIC) # Resize to 

        print(f'\nlabel = {label}')
        print(label)
        y = np.array([ 0 for _ in range(3)])
        if label==-1: y[1] = 1
        elif label==0: y[0] = 1
        else: y[2] = 1

        print(f'y = {y}\n') 

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        axs[0,0].contourf(data[1].T, 20, cmap='rainbow')
        axs[0,1].contourf(data[0].T, 20, cmap='rainbow')

        # levels for the phase angle to make it not interpolate 
        levels = MaxNLocator(nbins=10).tick_values(data[4].min(), data[4].max())
        cmap = plt.get_cmap('CMRmap')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        axs[1,0].pcolormesh(data[4].T, cmap=cmap, norm=norm)
        
        n=-1
        color_array = np.sqrt(((data[2]-n)/2)**2 + ((data[3]-n)/2)**2)
        axs[1,1].quiver(data[2].T, data[3].T, color_array, scale=2) # Plot vector field

        guiEvent, guiValues = gui.show_figure(fig)
        plt.close(fig)


def plot_training_data_rcnn():

    from PIL import Image

    npzPath = 'D:/Master/TTK-4900-Master/data/training_data/2016/rcnn/'

    with np.load(npzPath + 'data.npz', allow_pickle=True) as h5f:
        data = h5f['arr_0']
    with np.load(npzPath + 'labels.npz', allow_pickle=True) as h5f:
        labels = h5f['arr_0']
    with np.load(npzPath + 'box_idxs.npz', allow_pickle=True) as h5f:
        box_idxs = h5f['arr_0']

    for i, d in enumerate(data):

        width, height= len(d[:,:,0])*4, len(d[:,:,0][0])*4

        ssl = cv2.resize(d[:,:,0], dsize=(height, width), interpolation=cv2.INTER_CUBIC) 
        uvel = cv2.resize(d[:,:,1], dsize=(height, width), interpolation=cv2.INTER_CUBIC) 
        vvel = cv2.resize(d[:,:,2], dsize=(height, width), interpolation=cv2.INTER_CUBIC) 

        from sklearn.preprocessing import MinMaxScaler
        scaler = [MinMaxScaler(feature_range=(0,255)) for _ in range(3)]
        ssl_scaled = scaler[0].fit_transform(ssl)
        uvel_scaled = scaler[1].fit_transform(uvel)
        vvel_scaled = scaler[2].fit_transform(vvel)

        
        '''
        X = [ssl_scaled, uvel_scaled, vvel_scaled]
        nChannels = len(X)
        img = np.zeros((height, width, nChannels))
        for w in range(width):
            for h in range(height):
                for c in range(nChannels):
                    img[h,w,c] = X[c][w,h]
        
        fig, ax = plt.subplots()
        plt.contourf( img[:,:,0], cmap='rainbow', levels=100)
        n=-1
        color_array = np.sqrt(((img[:,:,1]-n)/2)**2 + ((img[:,:,2]-n)/2)**2)
        ax.quiver(img[:,:,1], img[:,:,2], color_array)#, scale=3)#, headwidth=0.5, width=0.01), #units="xy", ) # Plot vector field      
        '''

        img = np.asarray(Image.open(f'D:/master/TTK-4900-Master/keras-frcnn/train_images/{i}.png'))
        #ax.imshow(im/255, aspect="auto")
        print(img[-1,-1])
        #im[-1,]
        #plt.gca().invert_yaxis()
        #plt.contourf( im[:,:,0], cmap='rainbow', levels=100)
        
        '''
        for j, (box, label) in enumerate( zip(np.array(box_idxs[i]), np.array(labels[i])) ):
            e1, e2 = box.dot(4) # edge coordinates 
            size = np.subtract(e2,e1) # [width, height]

            if label == 1:
                rect = Rectangle(e1,size[0],size[1], edgecolor='b', facecolor="none")
            else:
                rect = Rectangle(e1,size[0],size[1], edgecolor='r', facecolor="none")
            ax.add_patch(rect)

        plt.show()
        '''

if __name__ == '__main__':
    plot_training_data()
    plot_training_data_rcnn()