import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle

npzPath = 'D:/Master/TTK-4900-Master/data/training_data/2016/rcnn/'

with np.load(npzPath + 'data.npz', allow_pickle=True) as h5f:
    data = h5f['arr_0']
with np.load(npzPath + 'labels.npz', allow_pickle=True) as h5f:
    labels = h5f['arr_0']
with np.load(npzPath + 'box_idxs.npz', allow_pickle=True) as h5f:
    box_idxs = h5f['arr_0']


#ssl = cv2.resize(data[0,:,:,0], None, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
#uvel = cv2.resize(data[0,:,:,1], None, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
#vvel = cv2.resize(data[0,:,:,2], None, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

ssl = data[0,:,:,0]
uvel = data[0,:,:,1]
vvel = data[0,:,:,2]

X = list(range(0,len(ssl)))
Y = list(range(0,len(ssl[0])))
X,Y = np.meshgrid(X,Y)

divUV = np.ufunc.reduce(np.add,np.gradient(uvel)) + np.ufunc.reduce(np.add,np.gradient(vvel))

'''
fig, ax = plt.subplots(1,2, figsize=(12, 8))
ax[0].streamplot(X,Y,uvel.T, vvel.T, density=6)#, color=divUV, cmap=plt.cm.RdBu)

#cm = ax[0].pcolormesh(divUV.T, cmap=plt.cm.RdBu, alpha=0.4)
cm = ax[0].contourf( ssl.T, cmap='rainbow', levels=40, alpha=0.5)
#fig.colorbar(cm)

ax[1].contourf( ssl.T, cmap='rainbow', levels=40)
n=-1
color_array = np.sqrt(((uvel-n)/2)**2 + ((vvel-n)/2)**2)
ax[1].quiver(uvel.T, vvel.T, color_array)#, scale=3)#, headwidth=0.5, width=0.01), #units="xy", ) # Plot vector field 

for j, (box, label) in enumerate( zip(np.array(box_idxs[0]), np.array(labels[0])) ):
    e1, e2 = box#.dot(4) # edge coordinates 
    size = np.subtract(e2,e1) # [width, height]

    if label == 1:
        rect1 = Rectangle(e1,size[0],size[1], edgecolor='b', facecolor="none",linewidth=2)
        rect2 = Rectangle(e1,size[0],size[1], edgecolor='b', facecolor="none")
    else:
        rect1 = Rectangle(e1,size[0],size[1], edgecolor='r', facecolor="none",linewidth=2)
        rect2 = Rectangle(e1,size[0],size[1], edgecolor='r', facecolor="none")
    ax[0].add_patch(rect1)
    ax[1].add_patch(rect2)
'''

fig, ax = plt.subplots(1,1, figsize=(12, 8))
ax.streamplot(X,Y,uvel.T, vvel.T, density=8)#, color=divUV, cmap=plt.cm.RdBu)
cm = ax.contourf( ssl.T, cmap='rainbow', levels=40, alpha=0.7)

for j, (box, label) in enumerate( zip(np.array(box_idxs[0]), np.array(labels[0])) ):
    e1, e2 = box#.dot(4) # edge coordinates 
    size = np.subtract(e2,e1) # [width, height]

    if label == 1:
        rect1 = Rectangle(e1,size[0],size[1], edgecolor='b', facecolor="none",linewidth=2)
    else:
        rect1 = Rectangle(e1,size[0],size[1], edgecolor='r', facecolor="none",linewidth=2)
    ax.add_patch(rect1)

plt.show()
