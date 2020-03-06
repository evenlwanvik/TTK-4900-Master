import numpy as np
import os
import zipfile
import scipy.io
import h5py



def h5_to_npz():
    dirpath = 'C:/Master/TTK-4900-Master/data/training_data/2016/h5/'
    zippath = dirpath + 'training_data.zip'
    savedir = 'C:/Master/TTK-4900-Master/data/training_data/2016/'

    with zipfile.ZipFile(zippath) as z:
        for fname in z.namelist():
            if not os.path.isdir(fname) and fname.endswith('.h5'):
                # read the file
                with z.open(fname) as f:
                    with h5py.File(f) as h5f:
                        print("Keys: %s" % f.keys())
                        a_group_key = list(f.keys())[0]

                        # Get the data
                        data = list(f[a_group_key])

                    if not os.path.exists(savedir):
                        np.savez_compressed( f'{savedir}/sst_train.npz', data[0])
                        np.savez_compressed( f'{savedir}/ssl_train.npz', data[1])
                        np.savez_compressed( f'{savedir}/uvel_train.npz', data[2])
                        np.savez_compressed( f'{savedir}/vvel_train.npz', data[3])
                        np.savez_compressed( f'{savedir}/phase_train.npz', data[4])


if __name__ == '__main__':
    h5_to_npz()