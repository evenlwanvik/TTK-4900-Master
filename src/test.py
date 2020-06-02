import numpy as np
import cv2
import matplotlib.pyplot as plt


fig, ax = plt.subplots(1,4)#,figsize=(10,12))

uvel_path = 'C:/Users/47415/Master/TTK-4900-Master/data/training_data/2016/uvel_train.npz'
vvel_path = 'C:/Users/47415/Master/TTK-4900-Master/data/training_data/2016/vvel_train.npz'
ssl_path = 'C:/Users/47415/Master/TTK-4900-Master/data/training_data/2016/ssl_train.npz'
sst_path = 'C:/Users/47415/Master/TTK-4900-Master/data/training_data/2016/sst_train.npz'

with np.load(uvel_path, allow_pickle=True) as data:
    x = data['arr_0'][:,0]
N = len(x)
y = np.zeros((N, 20, 20))
for i in range(N):
    y[i] = cv2.resize(x[i], dsize=(20, 20), interpolation=cv2.INTER_CUBIC) 
y = y.flatten()
hist, bins = np.histogram(y[np.isfinite(y)], 100)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
ax[0].bar(center, hist, align='center', width=width)

with np.load(vvel_path, allow_pickle=True) as data:
    x = data['arr_0'][:,0]
N = len(x)
y = np.zeros((N, 20, 20))
for i in range(N):
    y[i] = cv2.resize(x[i], dsize=(20, 20), interpolation=cv2.INTER_CUBIC) 
y = y.flatten()
hist, bins = np.histogram(y[np.isfinite(y)], 100)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
ax[1].bar(center, hist, align='center', width=width)

with np.load(ssl_path, allow_pickle=True) as data:
    x = data['arr_0'][:,0]
N = len(x)
y = np.zeros((N, 20, 20))
for i in range(N):
    y[i] = cv2.resize(x[i], dsize=(20, 20), interpolation=cv2.INTER_CUBIC) 
y = y.flatten()
hist, bins = np.histogram(y[np.isfinite(y)], 100)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
ax[2].bar(center, hist, align='center', width=width)

with np.load(sst_path, allow_pickle=True) as data:
    x = data['arr_0'][:,0]
N = len(x)
y = np.zeros((N, 20, 20))
for i in range(N):
    y[i] = cv2.resize(x[i], dsize=(20, 20), interpolation=cv2.INTER_CUBIC) 
y = y.flatten()
hist, bins = np.histogram(y[np.isfinite(y)], 100)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
ax[3].bar(center, hist, align='center', width=width)


ax[0].set_xlim([-0.5,0.5])
ax[1].set_xlim([-0.5,0.5])



plt.show()
