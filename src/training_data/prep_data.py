import numpy as np
import os
import zipfile
import scipy.io
import h5py


def h5_to_npz():
    dirpath = 'C:/Master/TTK-4900-Master/data/training_data/2016/h5/'
    zippath = dirpath + 'training_data.zip'
    savedir = 'C:/Master/TTK-4900-Master/data/training_data/2016/new/'

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

def count_labels():
    dirpath = 'C:/Master/TTK-4900-Master/data/training_data/2016/h5/'
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
    dirpath = 'C:/Master/TTK-4900-Master/data/training_data/2016/h5/'
    for fname in os.listdir(dirpath):
        if fname.endswith(".h5"):
            os.rename(dirpath+fname, dirpath+'ds2_'+fname)

                        
def common_npz():
    ''' Create a common npz file for all 4 measurements, kind of like RGB '''

    sst_path = 'C:/Master/TTK-4900-Master/data/training_data/2016/new/sst_train.npz'
    ssl_path = 'C:/Master/TTK-4900-Master/data/training_data/2016/new/ssl_train.npz'
    uvel_path = 'C:/Master/TTK-4900-Master/data/training_data/2016/new/uvel_train.npz'
    vvel_path = 'C:/Master/TTK-4900-Master/data/training_data/2016/new/vvel_train.npz'
    phase_path = 'C:/Master/TTK-4900-Master/data/training_data/2016/new/phase_train.npz'

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
            train[i].append([])
            for la in range(len(X[0][i][0])): 
                train[i][lo].append([])
                for c in range(nChannels):
                    train[i][lo][la].append(X[c][i][lo][la])
    
    savedir = 'C:/Master/TTK-4900-Master/data/training_data/2016/new/'

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    np.savez_compressed( f'{savedir}/full_train.npz', train)

if __name__ == '__main__':
    #h5_to_npz()
    #count_labels()
    #rename_files()
    common_npz()