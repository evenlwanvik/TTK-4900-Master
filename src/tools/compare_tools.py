import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model
import cv2

import importlib
import cnn 
from skimage import measure

from heapq import merge
from itertools import count
from matplotlib.patches import Rectangle
import pandas as pd
from scipy.interpolate import interp2d

def distance_column(x0, x, y0):
    # xstart, current col/row (width), ystart
    dist_x = (x - x0) ** 2
    yield dist_x, (x, y0) # distance from center, x and y idx
    for dy in count(1):
        dist = dist_x + dy ** 2
        yield dist, (x, y0 + dy)
        yield dist, (x, y0 - dy)

def circle_around(x0, y0, end_x):
    # array with labeled bin clusters, xstart, ystart
    for dist_point in merge(*(distance_column(x0, x, y0) for x in range(end_x))):
        yield dist_point

def find_labeled(labeled, ctr):
    # Start from center index and spiral outwards until we find first of a connected and labeled OW binary cluster
    # Find labeled cluster closest to center of eddy
    nx, ny = labeled.shape
    x0, y0 = ctr
    # Find max width of circle we will investigate
    maxWidth = min(x0, nx-x0, y0, ny-y0)*2
    #labeled_grid = labeled[x0-maxWidth:x0+maxWidth, y0-maxWidth:y0+maxWidth]
    for dist, (x, y) in itertools.islice(circle_around(x0, y0, maxWidth), maxWidth**2):
        if x > maxWidth-1 or y > maxWidth-1:
            return
        if labeled[x,y] > 0:
            return labeled[x,y], (x,y)       

def expand_cluster(OW_eddies_box, box, xy, edge, label):
    """ If we make it through the column/row, we don't have any more OW masks in this dimensions and we return True"""
    
    if xy == 'x':
        n = len(OW_eddies_box[0])
        getidx = lambda idx: (edge, idx)
    else:
        n = len(OW_eddies_box)
        getidx = lambda idx: (idx, edge)
    for i in range(n):
        val = OW_eddies_box[getidx(i)]
        if val == label:
            if xy == 'x': 
                if edge==0: box[0] -= 1; return False
                else:       box[2] += 1; return False
            else:        
                if edge==0: box[1] -= 1; return False
                else:       box[3] += 1; return False              
    return True

def include_full_cluster(labeled_array, box, label):
    """ labeled_array: Labels of binary clusters
        box: mutable list of box indexes passed by reference
        label: What label to include when increasing box"""
    
    #print(f"++Box shape before: {box}")
    for xy, edge in (('x', 0), ('x', -1), ('y', 0), ('y', -1)):
        i, maxit = 0, 300
        done = False
        # While we have more masks in this edge and dimension that are correct label -> continue to increase box size 
        while(not done):
            i += 1
            if i > maxit: return 0; print("Hit iteration marker, while loop most likely stuck")
            # If we are at edge dimensions of full array, just continue
            # We need to check if the current first or last dimension of col or row is larger or smaller than allowed
            box2d = {'x': [box[0], box[2]], 'y':[box[1], box[3]]}
            if box2d[xy][edge] in (0, labeled_array.shape[edge]): 
                print(f"dimension {xy} at edge {edge} is too large or small")
                break 
            labeled_box = labeled_array[box[0]:box[2], box[1]:box[3]]
            done = expand_cluster(labeled_box, box, xy, edge, label)
            #print(f"are we done?: [{done}], box: {box}")
    #print(f"++Box shape after: {box}")
    return 1

def get_min(data, b):
    """ Returns minimum value and index of box """
    data_box = data[b[0]:b[2], b[1]:b[3]]
    min_id = tuple(np.argwhere(data_box == np.min(data_box))[0])
    min_val = data_box[min_id]
    return min_id, min_val

def investigate_cluster(eddy_boxes, OW, OW_labeled, eddytype='cyclone'):
    """Check if the cluster is larger than the box boundaries and increase box size to encompass full cluster"""
    boxes_copy = np.zeros((len(eddy_boxes), 4), dtype=int)
    ctrIdxs = np.empty((len(eddy_boxes)), dtype=object)
    minOW = np.zeros(len(eddy_boxes))
    for boxId, b in enumerate(eddy_boxes):
        boxes_copy[boxId] = b[:]
        # Get index and value of minimum OW of the new box as center index
        ctrIdxs[boxId], minOW[boxId] = get_min(OW, boxes_copy[boxId])
        # Create array of the labeled OW clusters
        labeled_arr = OW_labeled[b[0]:b[2], b[1]:b[3]]
        # Get the label of the cluster at the minima, if no cluster present, delete from boxes
        label = labeled_arr[ctrIdxs[boxId]]
        print(f"Investigating {eddytype} nr {boxId} | label: {label} | box: {b} | center Idx: {ctrIdxs[boxId]}")
        if label <= 0:
            # TODO: Implement to delete array if needed
            print(f"No cluster in box {boxId}")
            continue
        # Increase the box size to encompass the whole labeled OW cluster
        if not include_full_cluster(OW_labeled, boxes_copy[boxId], label): 
            print("While loop stuck")
        # Count number of cells in cluster with same label
        # If box is changed, get the updated index and value of minimum OW of the new box as center index
        ctrIdxs[boxId], minOW[boxId] = get_min(OW, boxes_copy[boxId])
    print("")
    return ctrIdxs, minOW, boxes_copy

def plot_eddies(ax, eddy, color='r', numbered=True):
    for i, r in enumerate(eddy):
        rec = Rectangle((r[0], r[1]), r[2]-r[0], r[3]-r[1], edgecolor=color, linewidth=2, facecolor='none', zorder=10)
        ax.add_patch(rec)
        if numbered:
            ax.annotate(str(i+1), (int(r[0]+(r[2]-r[0])/2-12), int(r[1]+(r[3]-r[1])/2)-12), color='white', size=18)
        
def box2coords(x, y, boxes):
    """Convert box indeces to box coordinates"""
    boxcoords = []
    for i, box in enumerate(boxes):
        b = box[:]
        if b[2] >= len(x): b[2] = len(x)-1
        if b[3] >= len(y): b[3] = len(y)-1
        boxcoords.append([x[b[0]], y[b[1]], x[b[2]], y[b[3]]])
    return boxcoords

def create_image(fig, nx, ny):        
    im = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    im = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    imCopy = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
    imH, imW, _ = imCopy.shape # col, row
    winScaleW, winScaleH = imW*1.0/nx, imH*1.0/ny # Scalar coeff from dataset to cv2 image
    return im, winScaleW, winScaleH

def set_eddy_census(cellclusters, ctrIdxs, boxes, minOW, lon, lat, xc=[], yc=[], uvel=None, vvel=None, res=0.083, meastype='SAT'):

    census = np.zeros((len(boxes), 7))

    for i, b in enumerate(boxes):
        census[i, 0] = minOW[i]
        if meastype=='SINMOD':
            # Quick out of bounds check for box 
            if b[2] >= len(xc): b[2] = len(xc)-1
            if b[3] >= len(yc): b[3] = len(yc)-1
      
            lon_box = lon[b[0]:b[2]+1, b[1]:b[3]+1]
            lat_box = lat[b[0]:b[2]+1, b[1]:b[3]+1]  
            
            census[i, 1] = lon_box[ctrIdxs[i]]      # Lon center
            census[i, 2] = lat_box[ctrIdxs[i]]      # Lat center

            census[i, 4] = abs(xc[b[2]] - xc[b[0]]) / 1e3 # Width
            census[i, 5] = abs(yc[b[3]] - yc[b[1]]) / 1e3 # height
        else:
            lon_box = lon[b[0]:b[2]+1]
            lat_box = lat[b[1]:b[3]+1]
            census[i, 1] = lon_box[ctrIdxs[i][0]]
            census[i, 2] = lat_box[ctrIdxs[i][1]]
            census[i, 4] = 111.320e3 * abs(b[2]-b[0]) * np.cos(lat_box[ctrIdxs[i][1]]) * res / 1e3 
            census[i, 5] = 110.54e3 * abs(b[3]-b[1]) * res  / 1e3 # 0.083 degrees resolution per index 

        # Box with labeled clusters of binary mask given by OW threshold
        cell_cluster = cellclusters[b[0]:b[2]+1, b[1]:b[3]+1]
        # Get the label of this eddy cluster
        label = cell_cluster[ctrIdxs[i]] 
        # Find the average velocity of the eddy cells to give a measure for eddy circulation
        cluster_idxs = np.where(cell_cluster == label)
        nCells = len(cluster_idxs[0])
        census[i, 3] = nCells

        if uvel is not None or vvel is not None:
            cluster_uvel = uvel[cluster_idxs]
            cluster_vvel = vvel[cluster_idxs]
            census[i, 6] = np.sum(np.sqrt(np.square(cluster_uvel), np.square(cluster_vvel)))/nCells
        else: 
            census[i, 6] = 0.0

    name_list = ['minOW','lon[º]','lat[º]','cells','width[km]','height[km]','ciculation[m/s]']   
    df = pd.DataFrame(census, index= np.arange(1,len(boxes)+1), columns=name_list)

    return df

def deriv1_central_diff_2D(a,x,y):
    # Take the first derivative of a with respect to x and y using
    # centered central differences. The variable a is a 3D field.   
    nx,ny = a.shape
    dadx = np.zeros((nx,ny))                            
    dady = np.zeros((nx,ny))
    
    for j in range(0,ny):
        dadx[0,j] = (a[1,j] - a[0,j]) / (x[1,j] - x[0,j])
        for i in range(1,nx-1):
            dadx[i,j] = (a[i+1,j] - a[i-1,j]) / (x[i+1,j] - x[i-1,j])
        dadx[nx-1,j] = (a[nx-1,j] - a[nx-2,j]) / (x[nx-1,j] - x[nx-2,j])
    
    for i in range(0,nx):
        dady[i,0]=(a[i,1] - a[i,0]) / (y[i,1] - y[i,0])
        for j in range(1,ny-1):
            dady[i,j]=(a[i,j+1] - a[i,j-1]) / (y[i,j+1] - y[i,j-1])
        dady[i,ny-1]=(a[i,ny-1] - a[i,ny-2]) / (y[i,ny-1] - y[i,ny-2])
    return dadx,dady 

def calc_OW(lon,lat,uvel,vvel,OW_start):  

    ########################################################################
    # Initialize variables
    ########################################################################
    
    # Since they are masked arrays (in the mask, True = NaN value), we can fill the masked values with 0.0 to describe land
    uvel.set_fill_value(0.0)
    vvel.set_fill_value(0.0)

    # Create an ocean mask which has value True at ocean cells.
    ocean_mask = ~uvel.mask
    n_ocean_cells = uvel.count()
    
    nx,ny= uvel.shape
    
    # Compute cartesian distances for derivatives, in m
    R = 6378e3
    
    x = np.zeros((nx,ny))
    y = np.zeros((nx,ny))

    for i in range(0,nx):
        for j in range(0,ny):
            if lon.ndim > 1:
                x[i,j] = 2.*np.pi*R*np.cos(lat[i,j]*np.pi/180.)*lon[i,j]/360.
                y[i,j] = 2.*np.pi*R*lat[i,j]/360.
            else:
                x[i,j] = 2.*np.pi*R*np.cos(lat[j]*np.pi/180.)*lon[i]/360.
                y[i,j] = 2.*np.pi*R*lat[j]/360.
            
    # Gridcell area
    #dx,dy,grid_area = grid_cell_area(x,y)
    
    ########################################################################
    #  Compute Okubo-Weiss
    ########################################################################
    
    uvel = uvel.filled(0.0)
    vvel = vvel.filled(0.0)
    
    # velocity derivatives
    du_dx,du_dy = deriv1_central_diff_2D(uvel,x,y)
    dv_dx,dv_dy = deriv1_central_diff_2D(vvel,x,y)
    # strain and vorticity
    normal_strain = du_dx - dv_dy
    shear_strain = du_dy + dv_dx
    vorticity = dv_dx - du_dy
    
    # Compute OW, straight and then normalized with its standart deviation
    OW_raw = normal_strain ** 2 + shear_strain ** 2 - vorticity ** 2
    OW_mean = OW_raw.sum() / n_ocean_cells
    OW_std = np.sqrt(np.sum((np.multiply(ocean_mask,(OW_raw - OW_mean)) ** 2)) / n_ocean_cells)
    OW = OW_raw / OW_std
    
    # We create a mask with the possible location of eddies, meaning OW<-0.2
    OW_mask = np.zeros(OW.shape,dtype=int)
    OW_mask[np.where(OW < OW_start)] = 1

    # Seperate masks for cyclone and anti-cyclone depending on the vorticity polarity and magnitude     
    cyc_mask = np.where(vorticity < -1e-12, OW_mask, 0)
    acyc_mask = np.where(vorticity > 1e-12, OW_mask, 0)
    

    return OW, vorticity, OW_mask, cyc_mask, acyc_mask

def rotate_vector(u_east, v_north, lons, lon_displacement):
    # Current values in the file are stored as east-north components.
    # We need to read these and rotate into model currents:
    nx, ny = u_east.shape
    # Angle by which we need to rotate currents to get the oriented with the model grid.
    phi = -(np.pi/180)*(lon_displacement-lons)
    u_grid = np.ma.zeros((nx,ny))
    v_grid = np.ma.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            rotMat = np.array([[np.cos(phi[i,j]), -np.sin(phi[i,j])], [np.sin(phi[i,j]),np.cos(phi[i,j])]])
            vel_vec = np.array([[u_east[i,j]],[v_north[i,j]]])
            rotated = rotMat.dot(vel_vec)
            u_grid[i,j] = rotated[0]
            v_grid[i,j] = rotated[1]
    return np.ma.masked_invalid(u_grid), np.ma.masked_invalid(v_grid)


def interp2d_masked(a, x, y, m, kind='linear'):
    """ Wrapper 2d interpolation fucntion for a masked array
    Array, x, y, upsample factor """
    # Set masked areas to mean, we will be setting mask to NaN, but it won't be able to 
    zeromask = a[:]
    maskval = np.mean(a)
    #maskval = a.max()
    zeromask[np.isnan(zeromask)] = maskval
    x_intp = np.linspace(x.min(), x.max(), len(x)*m)
    y_intp = np.linspace(y.min(), y.max(), len(y)*m)
    f = interp2d(x, y, zeromask.T, kind=kind)
    return f(x_intp, y_intp).T # Transpose to return shape (x,y) 

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
class CentroidTracker():
	def __init__(self, maxDisappeared=2):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
			# return early as there are no centroids or tracking info
			# to update
			return self.objects