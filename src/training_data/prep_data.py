import numpy as np
import os
import zipfile
import scipy.io
import h5py
import cv2

def h5_to_npz_normal():
    dirpath = 'D:/Master/TTK-4900-Master/data/training_data/2016/h5/'
    zippath = dirpath + 'training_data.zip'
    savedir = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/'

    lon = []
    lat = []
    sst = []
    ssl = []
    uvel = []
    vvel = []
    phase = []

    with zipfile.ZipFile(zippath) as z:
        for fname in z.namelist():
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
    dirpath = 'D:/Master/TTK-4900-Master/data/training_data/2016/h5/rcnn/'
    zippath = dirpath + 'training_data.zip'
    savedir = 'D:/Master/TTK-4900-Master/data/training_data/2016/rcnn/'

    data = []
    box_idxs = []
    labels = []

    with zipfile.ZipFile(zippath) as z:
        for fname in z.namelist():
            if not os.path.isdir(fname) and fname.endswith('.h5'):
                # read the file
                with z.open(fname, 'r') as zf:
                    with h5py.File(zf, 'r') as hf:
                        data.append(hf['/data'][()].T)
                        box_idxs.append(hf['/box_idxs'][()])
                        labels.append(hf['/labels'][()].flatten())

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    np.savez_compressed( f'{savedir}/data.npz', data)
    np.savez_compressed( f'{savedir}/box_idxs.npz', box_idxs)
    np.savez_compressed( f'{savedir}/labels.npz', labels)


def prep_rcnn():

    from PIL import Image
    import pandas as pd

    npzPath = 'D:/Master/TTK-4900-Master/data/training_data/2016/rcnn/'

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
     
        im = Image.fromarray(img.astype('uint8'), mode='RGB')
        #new_im = im.resize((1000,600),Image.BICUBIC)
        im.save(f'D:/master/TTK-4900-Master/keras-frcnn/train_images/{i}.png')

        #### add xmin, ymin, xmax, ymax and class as per the format required ####

        

        for j, (box, label) in enumerate( zip(np.array(box_idxs[i]), np.array(labels[i])) ):
            e1, e2 = box#.dot(4) # edge coordinates, also needs to be rescaled
            size = np.subtract(e2,e1) # [width, height]

            # Create a row for each bounding box for a given image
            if label == 1:
                input_str_arr.append(f'train_images/{i}.png'+','+str(int(e1[0]))+','+str(int(e1[1]))+','+str(int(e2[0]))+','+str(int(e2[1]))+','+'anti-cyclone')
            else: 
                input_str_arr.append(f'train_images/{i}.png'+','+str(int(e1[0]))+','+str(int(e1[1]))+','+str(int(e2[0]))+','+str(int(e2[1]))+','+'cyclone')
    
    X = pd.DataFrame()
    X['format'] = input_str_arr
    X.to_csv('D:/master/TTK-4900-Master/keras-frcnn/annotate.txt', header=None, index=None, sep=' ')


def count_labels():
    dirpath = 'D:/Master/TTK-4900-Master/data/training_data/2016/h5/'
    zippath = dirpath + 'training_data.zip'

    from collections import Counter

    label = []
    with zipfile.ZipFile(zippath) as z:
        for fname in z.namelist():
            if not os.path.isdir(fname) and fname.endswith('.h5'):
                # read the file
                with z.open(fname, 'r') as zf:
                    with h5py.File(zf, 'r') as hf:
                        label.append(int(hf['/label'][()]))

    print(Counter(label))


def rename_files():
    dirpath = 'D:/Master/TTK-4900-Master/data/training_data/2016/h5/'
    for fname in os.listdir(dirpath):
        if fname.endswith(".h5"):
            os.rename(dirpath+fname, dirpath+'ds2_'+fname)

                        
def common_npz():
    ''' Create a common npz file for all 4 measurements, kind of like RGB '''

    sst_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/sst_train.npz'
    ssl_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/ssl_train.npz'
    uvel_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/uvel_train.npz'
    vvel_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/vvel_train.npz'
    phase_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/phase_train.npz'

    X = []

    with np.load(ssl_path, allow_pickle=True) as data:
        X.append(data['arr_0'][:,0])
        Y = data['arr_0'][:,1]
    with np.load(uvel_path, allow_pickle=True) as data:
        X.append(data['arr_0'][:,0])
    with np.load(vvel_path, allow_pickle=True) as data:
        X.append(data['arr_0'][:,0])
        Y = data['arr_0'][:,1]
    with np.load(phase_path, allow_pickle=True) as data:
        X.append(data['arr_0'][:,0]) 
        Y = data['arr_0'][:,1]       
   
    nTeddies = len(X[0])
    nChannels = len(X)
    train = []

    for i in range(nTeddies): # Eddies
        train.append([])
        for lo in range(len(X[0][i])): 
            +[i].append([])
            for la in range(len(X[0][i][0])): 
                train[i][lo].append([])
                for c in range(nChannels):
                    train[i][lo][la].append(X[c][i][lo][la])
    
    savedir = 'D:/Master/TTK-4900-Master/data/training_data/2016/new/'

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    np.savez_compressed( f'{savedir}/full_train.npz', train)


def yml2xml_annotate():

    import xmlplain
    import copy

    npzPath = 'D:/Master/TTK-4900-Master/data/training_data/2016/rcnn/'


    # Read the YAML file
    with open("D:/master/TTK-4900-Master/my_mrcnn/xml/template.yml") as inf:
        root = xmlplain.obj_from_yaml(inf)

    with np.load(npzPath + 'data.npz', allow_pickle=True) as h5f:
        nSamples = len(h5f['arr_0'])
        width, height, depth = h5f['arr_0'][0].shape
    with np.load(npzPath + 'labels.npz', allow_pickle=True) as h5f:
        labels = h5f['arr_0']
    with np.load(npzPath + 'box_idxs.npz', allow_pickle=True) as h5f:
        box_idxs = h5f['arr_0']

    for i in range(nSamples):

        root['annotation']['filename'] = f'{i}.png'
        root['annotation']['size']['width'] = width
        root['annotation']['size']['height'] = height
        root['annotation']['size']['depth'] = depth

        for j, (box, label) in enumerate( zip(np.array(box_idxs[i]), np.array(labels[i])) ):
            e1, e2 = box#.dot(4) # edge coordinates, also needs to be rescaled

            if j is not 0: 
                obj_copy = copy.deepcopy(root['annotation']['object'])
                root['annotation'].append(obj_copy)

            if label == 1: root['annotation']['object']['name'] = 'anti-cyclone'
            else: root['annotation']['object']['name'] = 'cyclone'
            root['annotation']['object']['bndbox']['xmin'] = e1[0]
            root['annotation']['object']['bndbox']['ymin'] = e1[1]
            root['annotation']['object']['bndbox']['xmax'] = e2[0]
            root['annotation']['object']['bndbox']['ymax'] = e2[1]

        out_path = f'D:/master/TTK-4900-Master/my_mrcnn/xml/{i}.xml'
        # Output back XML
        with open(out_path, 'w') as outf:
            xmlplain.xml_from_obj(root, outf, pretty=True)

def xml_annotate():
    import xml.dom.minidom as xdm

    doc = xdm.parse('D:/master/TTK-4900-Master/my_mrcnn/xml/template.xml')

    print(doc.nodeName)
    
    print(len(doc.getElementsByTagName("annotation")))

if __name__ == '__main__':
    #h5_to_npz()
    #prep_rcnn()
    #yml2xml_annotate()
    xml_annotate()
    #h5_to_npz_rcnn()
    #count_labels()
    #rename_files()
    #common_npz()