import numpy as np
import os
import zipfile
import scipy.io
import h5py
import zipfile


def h5_to_npz():
    dirpath = 'C:/Master/TTK-4900-Master/data/training_data/2016/h5/'
    zippath = dirpath + 'training_data.zip'
    savedir = 'C:/Master/TTK-4900-Master/data/training_data/2016/new'

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
    else:
        with np.load(f'{savedir}/lon.npz', 'w+', allow_pickle=True) as data:
            np.savez_compressed( f'{savedir}/lon.npz', np.append(data['arr_0'], lon, axis=0))
        with np.load(f'{savedir}/lat.npz', 'w+', allow_pickle=True) as data:
            np.savez_compressed( f'{savedir}/lat.npz', np.append(data['arr_0'], lat, axis=0))
        with np.load(f'{savedir}/sst_train.npz', 'w+', allow_pickle=True) as data:
            np.savez_compressed( f'{savedir}/sst_train.npz', np.append(data['arr_0'], sst, axis=0))
        with np.load(f'{savedir}/ssl_train.npz', 'w+', allow_pickle=True) as data:
            np.savez_compressed( f'{savedir}/ssl_train.npz', np.append(data['arr_0'], ssl, axis=0))
        with np.load(f'{savedir}/uvel_train.npz', 'w+', allow_pickle=True) as data:
            np.savez_compressed(f'{savedir}/uvel_train.npz', np.append(data['arr_0'], uvel, axis=0))
        with np.load(f'{savedir}/vvel_train.npz', 'w+', allow_pickle=True) as data:
            np.savez_compressed(f'{savedir}/vvel_train.npz', np.append(data['arr_0'], vvel, axis=0))
        with np.load(f'{savedir}/phase_train.npz', 'w+', allow_pickle=True) as data:
            np.savez_compressed(f'{savedir}/phase_train.npz', np.append(data['arr_0'], phase, axis=0))
                       

if __name__ == '__main__':
    h5_to_npz()