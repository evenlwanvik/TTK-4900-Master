import numpy as np
import os
import zipfile
import scipy.io 
import h5py
import cv2

def h5_to_npz_normal():
    """ Convert the h5 training samples from the compressed folder created by the MATLAB 
    training data collection application. """
    dirpath = 'C:/Users/47415/Master/TTK-4900-Master/data/h5/'
    zippath = dirpath + 'training_data.zip'
    savedir = 'C:/Users/47415/Master/TTK-4900-Master/data/'

    lon = []
    lat = []
    sst = []
    ssl = []
    uvel = []
    vvel = []
    phase = []

    # Training samples found to be incorrectly labeled or just bad training samples, 
    # e.g., background samples containing circulatory ocean current
    #excludelist = [None]
    excludelist = [4, 5, 27, 36, 56, 61, 68, 141, 163, 168, 169, 171, 176, 179, 183, 248, 250,
                254, 256, 277, 289, 299, 306, 316, 321, 390, 416, 422, 431, 440, 465, 470, 563, 589,
                687, 697, 992, 1006, 1010, 1011, 1013, 1024, 1048, 1060, 1072, 1159, 1160, 1161, 
                1163, 1191, 1202, 1220, 1229, 1249, 1275, 1284, 1298, 1504, 1509, 1511, 1514, 1567, 1735] # (+1 Python iter)



    with zipfile.ZipFile(zippath) as z:
        for i, fname in enumerate(z.namelist()):
            if not i in excludelist:
                if not os.path.isdir(fname) and fname.endswith('.h5'):
                    # read the file
                    with z.open(fname, 'r') as zf:
                        with h5py.File(zf, 'r') as hf:

                            lon.append([hf['/coordinates/lon'][()], int(hf['/label'][()])])
                            lat.append([hf['/coordinates/lat'][()], int(hf['/label'][()])])
                            sst.append([hf['/data/sst'][()], int(hf['/label'][()])])
                            ssl.append([hf['/data/ssl'][()], int(hf['/label'][()])])
                            uvel.append([hf['/data/uvel'][()], int(hf['/label'][()])])
                            vvel.append([hf['/data/vvel'][()], int(hf['/label'][()])])
                            phase.append([hf['/data/phase'][()], int(hf['/label'][()])])

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    np.savez_compressed( f'{savedir}/lon.npz', lon)
    np.savez_compressed( f'{savedir}/lat.npz', lat)
    np.savez_compressed( f'{savedir}/sst_train.npz', sst)
    np.savez_compressed( f'{savedir}/ssl_train.npz', ssl)
    np.savez_compressed( f'{savedir}/uvel_train.npz', uvel)
    np.savez_compressed( f'{savedir}/vvel_train.npz', vvel)
    np.savez_compressed( f'{savedir}/phase_train.npz', phase)

def h5_to_npz_rcnn():
    """ Same as above, just for the Faster R-CNN training data """
    dirpath = 'D:/Master/TTK-4900-Master/data/h5/rcnn/'
    zippath = dirpath + 'training_data.zip'
    savedir = 'D:/Master/TTK-4900-Master/data/rcnn/'

    data = []
    box_idxs = []
    labels = []

    with zipfile.ZipFile(zippath) as z:
        for fname in z.namelist():
            if not os.path.isdir(fname) and fname.endswith('.h5'):
                # read the file
                with z.open(fname, 'r') as zf:
                    with h5py.File(zf, 'r') as hf:
                        print(fname)
                        print(hf.keys())
                        data.append(hf['/data'][()].T)
                        box_idxs.append(hf['/box_idxs'][()])
                        labels.append(hf['/labels'][()].flatten())

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    np.savez_compressed( f'{savedir}/data.npz', data)
    np.savez_compressed( f'{savedir}/box_idxs.npz', box_idxs)
    np.savez_compressed( f'{savedir}/labels.npz', labels)


def prep_rcnn():
    """ For storing the Faster R-CNN npz training data as image and annotation text file """

    from PIL import Image
    import pandas as pd

    npzPath = 'D:/Master/TTK-4900-Master/data/rcnn/'

    with np.load(npzPath + 'data.npz', allow_pickle=True) as h5f:
        data = h5f['arr_0']
    with np.load(npzPath + 'labels.npz', allow_pickle=True) as h5f:
        labels = h5f['arr_0']
    with np.load(npzPath + 'box_idxs.npz', allow_pickle=True) as h5f:
        box_idxs = h5f['arr_0']

    input_str_arr = []

    for i, d in enumerate(data):

        # upscale image, we will be cropping for training
        width, height= len(d[:,:,0]), len(d[:,:,0][0])


        ################# Upscalem encode and save data as image #################

        ssl = d[:,:,0] #cv2.resize(d[:,:,0], dsize=(height, width), interpolation=cv2.INTER_CUBIC) 
        uvel = d[:,:,1] #cv2.resize(d[:,:,1], dsize=(height, width), interpolation=cv2.INTER_CUBIC) 
        vvel = d[:,:,2] #cv2.resize(d[:,:,2], dsize=(height, width), interpolation=cv2.INTER_CUBIC) 

        from sklearn.preprocessing import MinMaxScaler
        scaler = [MinMaxScaler(feature_range=(0,255)) for _ in range(3)]
        ssl_scaled = scaler[0].fit_transform(ssl)
        uvel_scaled = scaler[1].fit_transform(uvel)
        vvel_scaled = scaler[2].fit_transform(vvel)

        data_ensemble = [ssl_scaled, uvel_scaled, vvel_scaled]
        nChannels = len(data_ensemble)
        img = np.zeros((height, width, nChannels))
        for w in range(width):
            for h in range(height):
                for c in range(nChannels):
                    img[h,w,c] = data_ensemble[c][w,h]
     
        img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        im = Image.fromarray(img.astype('uint8'), mode='RGB')
        #new_im = im.resize((1000,600),Image.BICUBIC)
        im.save(f'D:/master/TTK-4900-Master/keras-frcnn/train_images/{i}.png')

        #### add xmin, ymin, xmax, ymax and class as per the format required ####

        

        for j, (box, label) in enumerate( zip(np.array(box_idxs[i]), np.array(labels[i])) ):
            e1, e2 = box.dot(5) # edge coordinates, also needs to be rescaled
            size = np.subtract(e2,e1) # [width, height]

            # Create a row for each bounding box for a given image
            if label == 1:
                input_str_arr.append(f'train_images/{i}.png'+','+str(int(e1[0]))+','+str(int(e1[1]))+','+str(int(e2[0]))+','+str(int(e2[1]))+','+'anti-cyclone')
            else: 
                input_str_arr.append(f'train_images/{i}.png'+','+str(int(e1[0]))+','+str(int(e1[1]))+','+str(int(e2[0]))+','+str(int(e2[1]))+','+'cyclone')
    
    X = pd.DataFrame()
    X['format'] = input_str_arr
    X.to_csv('D:/master/TTK-4900-Master/keras-frcnn/annotate.txt', header=None, index=None, sep=' ')

def rcnn_test_image():
    """ Show Faster R-CNN RGB encoded image """

    nc_fpath='D:/Master/data/cmems_data/global_10km/noland/smaller/phys_noland_2016_001.nc'
    ds = xr.open_dataset(nc_fpath)

    # Mask NaN - indicating land
    ssl = np.ma.masked_invalid(ds.zos[0].T)
    uvel = np.ma.masked_invalid(ds.uo[0,0].T)
    uvel = np.ma.masked_invalid(ds.uv[0,0].T)

    im = Image.fromarray(img.astype('uint8'), mode='RGB')
    #new_im = im.resize((1000,600),Image.BICUBIC)
    im.save(f'D:/master/TTK-4900-Master/keras-frcnn/train_images/{i}.png')

def count_labels():
    """ Count the number of classes in training data """
    dirpath = 'D:/Master/TTK-4900-Master/data/new/ssl_train.npz'

    from collections import Counter

    labels = []

    with np.load(dirpath, allow_pickle=True) as h5f:
        data = h5f['arr_0'][:,1]
        for label in data:
            labels.append(label)

    print(Counter(labels))



if __name__ == '__main__':
    #h5_to_npz_normal()
    #prep_rcnn()
    #yml2xml_annotate()
    #xml_annotate()
    #count_labels()
    #h5_to_npz_rcnn()
    #prep_rcnn()