import numpy as np
import matplotlib.pyplot as plt

data_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/rcnn/data.npz'
labels_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/rcnn/box_idxs.npz'
box_idxs_path = 'D:/Master/TTK-4900-Master/data/training_data/2016/rcnn/labels.npz'
model_fpath = 'D:/master/models/2016/rcnn_model.h5'
scaler_fpath = "D:/master/models/2016/rcnn_norm_scaler.pkl"
#2016/new
#200_days_2018

#scaler = [StandardScaler() for _ in range(nChannels)]
#scaler = MinMaxScaler(feature_range=(-1,1))
#probLim = 0.97

def train_model():

    with np.load(data_path, allow_pickle=True) as h5f:
        data = h5f['arr_0']
    with np.load(box_idxs_path, allow_pickle=True) as h5f:
        labels = h5f['arr_0']
    with np.load(labels_path, allow_pickle=True) as h5f:
        box_idxs = h5f['arr_0']

    nSamples, _, nChannels, height, width = data.shape


    X = np.zeros((nSamples, width, height, nChannels))
    boxes = np.zeros((nSamples, 4))

    for i,d in enumerate(data): # sample
        for w in range(width): # Row
            for h in range(height): # Column
                for c in range(nChannels): # Channels
                    X[i,w,h,c] = d[0,c,h,w]

        for j, (box, label) in enumerate( zip(np.array(box_idxs[i,0]), np.array(labels[i,0])) ):
            e1, e2 = box # edge coordinates 
            x1, y1 = e1
            x2, y2 = e2
            boxes[i] = [x1, y1, x2, y2]
        if i==3: break

    from PIL import Image 
    with Image 
    print(boxes, width, height)


    # add dataset name, image name, path to file, and its annotation
    self.add_image('eddydetection', image_id='00001', path='kangaroo/images/00001.jpg', 
                    annotation='kangaroo/annots/00001.xml')



# class that defines and loads the eddy dataset
class EddyDataset(utils.Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir):
        # define one class
        # define one class with dataset name, label, and class name
        self.add_class("eddydetection", 1, "cyclone")
        self.add_class("eddydetection", 2, "anti-cyclone")
        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        # find all images
        for filename in listdir(images_dir):
            # extract image id, i.e. only name, not the extension 
            image_id = filename[:-4]
            npz_path = images_dir + filename
            label_path = annotations_dir + 'labels.npz'
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=npz_path, annotation=label_path)
 
	# load the masks for an image
	def load_mask(self, image_id):
		# ...
 
	# load an image reference
	def image_reference(self, image_id):
		# ..
'''

if __name__ == '__main__':
    train_model()