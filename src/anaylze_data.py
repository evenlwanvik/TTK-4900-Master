from tools import dim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from tools import gui
import cv2

def plot_training_data():
    ''' Just to analyze some of the npz training data before generating more training data '''
    dirpath = 'C:/Master/TTK-4900-Master/data/training_data/2016/new/'

    sst = []
    ssl = []
    uvel = []
    vvel = []
    phase = []

    
    with np.load(dirpath+'ssl_train.npz', allow_pickle=True) as h5f:
        # The standard grid size we will use
        gridSize = dim.find_avg_dim(h5f['arr_0'][:,0])
        nLon = gridSize[0]
        nLat = gridSize[1]

    nTeddies = 10 # number of training eddies to analyze
    for i in range(nTeddies):
        data = []
        for j, fpath in enumerate((dirpath+'sst_train.npz', dirpath+'ssl_train.npz', dirpath+'uvel_train.npz', dirpath                               +'vvel_train.npz', dirpath+'phase_train.npz')):
            with np.load(fpath, allow_pickle=True) as h5f:
                data.append(h5f['arr_0'][i,0])


            data[j] = np.array(data[j], dtype='float32') # convert to numpy array
            data[j] = cv2.resize(data[j], dsize=(nLat, nLon), interpolation=cv2.INTER_CUBIC) # Resize to 

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


if __name__ == '__main__':
    plot_training_data()