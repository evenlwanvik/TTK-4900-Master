from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def main():



    npzPath = 'D:/Master/TTK-4900-Master/data/training_data/2016/rcnn/'

    with np.load(npzPath + 'data.npz', allow_pickle=True) as h5f:
        data = h5f['arr_0']
    with np.load(npzPath + 'labels.npz', allow_pickle=True) as h5f:
        labels = h5f['arr_0']
    with np.load(npzPath + 'box_idxs.npz', allow_pickle=True) as h5f:
        box_idxs = h5f['arr_0']

    for i in (13,14):
        fig, ax = plt.subplots()
        ax.contourf( data[i,:,:,0].T, cmap='rainbow', levels=100)
        n=-1
        color_array = np.sqrt(((data[i,:,:,1].T-n)/2)**2 + ((data[i,:,:,2].T-n)/2)**2)
        ax.quiver(data[i,:,:,1].T, data[i,:,:,2].T, color_array)#, scale=3)#, headwidth=0.5, width=0.01), #units="xy", ) # Plot vector field 

        for j, (box, label) in enumerate( zip(np.array(box_idxs[i]), np.array(labels[i])) ):
            e1, e2 = box#.dot(4) # edge coordinates 
            size = np.subtract(e2,e1) # [width, height]

            if label == 1:
                rect = Rectangle(e1,size[0],size[1], edgecolor='b', facecolor="none")
            else:
                rect = Rectangle(e1,size[0],size[1], edgecolor='r', facecolor="none")
            ax.add_patch(rect)

        plt.gca()#.invert_yaxis()

        plt.show()

if __name__ == '__main__':

    main()