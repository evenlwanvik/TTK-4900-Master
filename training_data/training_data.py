from eddies import load_netcdf4,eddy_detection,dataframe_eddies,plot_eddies,julianh2gregorian
from matplotlib.patches import Rectangle
from numpy import savez_compressed
import matplotlib.pyplot as plt
from gui import show_figure
from tools.bfs import bfs
from math import cos, pi
from operator import eq
import xarray as xr
import numpy as np
import argparse
import datetime
import random
import cv2
import os
import io
import sys

from noneddy import get_noneddies

argp = argparse.ArgumentParser()
argp.add_argument("-fp", "--fpath", default='C:/Master/data/cmems_data/global_10km/phys_noland_001.nc', help="rectangular patche size multiplier")
argp.add_argument("-rs", "--size", default=1.5, help="rectangular patche size multiplier")
argp.add_argument("-sd", "--savedir", default='C:/Master/TTK-4900-Master/training_data/data/', help="training data save dir")

def lon2km(lon, lat):
    """ Convert from longitudinal displacement to km """
    return lon * 111.320e3 * cos(lat)
    

def lat2km(lat):
    """ Convert from latitudinal displacement to km """
    return 110.54e3 * lat


def dim(a):
    """" Recursively add length of subdimensions to list """
    if not isinstance(a,(list,np.ndarray)):
        return []
    return [len(a)] + dim(a[0])


def find_max_dim(a):
    """" Find the largest dimension of list or array """
    return max( [dim(i) for i in a] )


def eddy_metrics(eddies_ma, centerIdxs, lon, lat):
    """ 
    Finds metrics such as index centroid and diameter of eddy 

    Parameters:
    ----------
    eddies_ma : masked array
        masked array received from the OW-R2 eddy_detection algorithm
    centereIdxs : tuple
        tuple with (lon,lat) indeces of the center coordinates of eddy
    
    returns:
    ----------
    float: diameter of the eddy
    tuple: (lon,lat) index of center
    """
    start = centerIdxs[0], centerIdxs[1]
    neighbors = (-1, 0), (0, +1), (+1, 0), (0, -1) # possible neighbors
    similar = eq # Eq method test the equality of the values.

    # Run BFS to find the indexes in eddy from the masked array
    eddyIdxs = np.array( list( bfs(eddies_ma, neighbors, start, similar) ) )

    # Find center lon/lat index of eddy
    lonCtrIdx = int( eddyIdxs[:,0].mean() )
    latCtrIdx = int( eddyIdxs[:,1].mean() )

    # Find the displacement in lon/lat direction in km, and use the largest as diameter of eddy.
    lonDiameter_km = lon2km( eddyIdxs[:,0].max()-eddyIdxs[:,0].min(),  lat[latCtrIdx]) * 0.083 
    latDiameter_km = lat2km( eddyIdxs[:,1].max()-eddyIdxs[:,1].min() ) * 0.083 # 0.083 degrees resolution per index

    largest_diameter_km = np.max([lonDiameter_km, latDiameter_km])

    return largest_diameter_km, (lonCtrIdx, latCtrIdx)


def check_cyclone(flag):
    # positive 1 denotes positive rotation, which is a cyclone in the norther hemisphere
    if   flag==1:  return "cyclone"
    elif flag==-1: return "anticyclone"
    else:          return "nothing"


def main():

    args, leftovers = argp.parse_known_args()

    print("\n--- loading netcdf")

    # load data
    (ds,t,lon,lat,depth,uvel_full,vvel_full,sst_full,ssl_full) =  load_netcdf4(args.fpath)

    # Confidence level, usually 90%
    R2_criterion = 0.9

    # OW value at which to begin the evaluation of R2, default was -1, want to use -8 to be absolutely sure
    OW_start = -8.0

    # Number of local minima to evaluate using R2 method.
    # Set low (like 20) to see a few R2 eddies quickly.
    # Set high (like 1e5) to find all eddies in domain.
    max_evaluation_points = 2000 #set to 2000 to capture avery minima, there should be around 1800

    # Minimum number of cells required to be identified as an eddie.
    min_eddie_cells = 2 # set to 3 to be coherent with the use of the R2 method, 3 points seems like a reasonable minimun for a correlation 

    # z-level to plot.  Usually set to 0 for the surface.
    k_plot = 0

    # Initializing lists for storing training data, format: [data(lon,lat),label]. 
    # TODO: Change this to make it more scalable?
    sst_train = []
    ssl_train = []
    uvel_train = []
    vvel_train = []
    phase_train = []
    nDataset = 5

    # Create eddy images for each day in datase
    #for day, time in enumerate(t):
    for day in range(2): # Run through code quickly 

        dateStr = "{:%d-%m-%Y}".format(datetime.date(1950, 1, 1) + datetime.timedelta(hours=float(t[day])) )
        print(f"--- Creating images for dataset {dateStr}")

        # create a text trap
        text_trap = io.StringIO()
        sys.stdout = text_trap

        # Run the OW-R2 algorithm
        lon,lat,u,v,vorticity,OW,OW_eddies,eddie_census,nEddies,circulation_mask = eddy_detection(
                lon,lat,depth,uvel_full,vvel_full,day,R2_criterion,OW_start,max_evaluation_points,min_eddie_cells)

        # restore stdout
        sys.stdout = sys.__stdout__

        # Also create mask of where non-eddies are
        OW_noneddies = np.zeros(OW.shape,dtype=int)
        OW_noneddies[np.where(OW > 0)] = 1 # non-eddy are as 1
        OW_noneddies = np.ma.masked_equal(OW_noneddies, 0) # rest is masked

        # =========================================================
        # ============== Prepare datasets and lists ===============
        # =========================================================

        eddyCtrIdx = []
        for i in range(0,nEddies):
            lonIdx = np.argmax(lon>eddie_census[2,i])-1
            latIdx = np.argmax(lat>eddie_census[3,i])-1
            eddyCtrIdx.append( (lonIdx, latIdx) )

        sst = sst_full[0,:,:].T
        ssl = ssl_full[0,:,:].T
        uvel = uvel_full[0,:,:].T
        vvel = vvel_full[0,:,:].T

        # =========================================================
        # ======= Create rectangular patches around eddies ========
        # =========================================================

        print(f"---++ Creating rectangles for {nEddies} eddies")

        savedImgCounter = 0 # saved image counter for file ID
        for eddyId, ctrIdx in enumerate(eddyCtrIdx): # nEddies

            ctrCoord = lon[ctrIdx[0]], lat[ctrIdx[1]]
            diameter_km = eddie_census[5][i] * 1000 # eddie_census is diameter in km

            bfs_diameter_km, bfs_center = eddy_metrics(OW_eddies, ctrIdx, lon, lat)

            print(f"---++++ Creating rectangles for eddy {eddyId} with center {ctrCoord} and diameter {diameter_km}")
            
            # Find rectangle metrics
            height = args.size * abs(diameter_km / 110.54e3) # 1 deg = 110.54 km, 1.2 to be sure the image covers the eddy
            width = args.size * abs(diameter_km / (111.320e3 * cos(lat[ctrIdx[1]]))) # 1 deg = 111.320*cos(latitude) km, using center latitude as ref
            
            lon_bnds = ctrCoord[0]-width/2.0, ctrCoord[0]+width/2.0
            lat_bnds = ctrCoord[1]-height/2.0, ctrCoord[1]+height/2.0
            
            # Indeces of current eddy image
            lonIdxs = np.where((lon > lon_bnds[0]) & (lon < lon_bnds[1]))[0]
            latIdxs = np.where((lat > lat_bnds[0]) & (lat < lat_bnds[1]))[0]
            
            sst_eddy   =   np.zeros((lonIdxs.size,latIdxs.size))
            ssl_eddy   =   np.zeros((lonIdxs.size,latIdxs.size))
            uvel_eddy  =   np.zeros((lonIdxs.size,latIdxs.size))
            vvel_eddy  =   np.zeros((lonIdxs.size,latIdxs.size))
            phase_eddy =   np.zeros((lonIdxs.size,latIdxs.size))

            # Positive rotation (counter-clockwise) is a cyclone in the northern hemisphere because of the coriolis effect
            if (eddie_census[1][i] > 0.0): cyclone = 1 # 1 is a cyclone, 0 is nothing and -1 is anti-cyclone (negative rotation)
            else: cyclone = -1
            
            for j, lo in enumerate(lonIdxs):
                for k, la in enumerate(latIdxs):
                    #idx = j*latIdxs.size + k
                    sst_eddy[j,k] = sst[lo,la]
                    ssl_eddy[j,k] = ssl[lo,la]
                    uvel_eddy[j,k] = uvel[lo,la]
                    vvel_eddy[j,k] = vvel[lo,la]
                    # Calculate the phase angle (direction) of the current
                    with np.errstate(all='ignore'): # Disable zero div warning
                        phase_eddy[j,k] = xr.ufuncs.rad2deg( xr.ufuncs.arctan2(vvel[lo,la], uvel[lo,la]) ) + 180  

            # =========================================================
            # ======= Create images of the rectangular patches ========
            # =========================================================

            fig, axs = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey=True)
            lo = lon[lonIdxs]
            la = lat[latIdxs]

            axs[0,0].contourf(lo, la, sst_eddy.T, 10, cmap='rainbow')
            axs[0,1].contourf(lo, la, ssl_eddy.T, 10, cmap='rainbow')
            axs[0,2].contourf(lo, la, uvel_eddy.T, 10, cmap='rainbow')
            axs[1,0].contourf(lo, la, vvel_eddy.T, 10, cmap='rainbow')
            axs[1,1].contourf(lo, la, phase_eddy.T, 10, cmap='CMRmap')
            title = dateStr + "_" + check_cyclone(cyclone)
            fig.suptitle(title, fontsize=16)
 
            # =========================================================
            # === Use simple GUI for selecting correct annoted data ===
            # =========================================================

            #guiEvent, guiValues = show_figure(fig)
            guiEvent = 'Yes' # Omit GUI selection
            if guiEvent=='Yes':
                # Create images?
                '''
                dirPath = 'C:/Master/TTK-4900-Master/images/'+dateStr+'/'
                if not os.path.exists(dirPath):
                    os.makedirs(dirPath)
                savedImgCounter = savedImgCounter + 1 
                imPath = dirPath + title + f"_{savedImgCounter}.png"   
                plt.savefig(imPath, bbox_inches='tight')
                '''

                sst_train.append([sst_eddy, cyclone]) # [data, label]
                ssl_train.append([ssl_eddy, cyclone]) 
                uvel_train.append([uvel_eddy, cyclone]) 
                vvel_train.append([vvel_eddy, cyclone]) 
                phase_train.append([phase_eddy, cyclone]) 

                print(f"---++++++ Saving image {eddyId}")   

            elif guiEvent=='No': 
                print(f"---++++++ Discarding image {eddyId}")
            
            plt.close(fig)


        # For each eddy, find a non-eddy!
        # Find all possible noneddies, then find all viable options and find randomly select n images for each day. We only need to find for one

        # We will create 100 more images than eddies that have been saved, from which the same amount of noneddies will be selected randomly from viable grids
        nGrids = savedImgCounter*100

        # Just use the largest dimension in training data so far, we will interpolate all grids that are not the biggest at the end anyways
        grid_size = find_max_dim(sst_train[0])

        subgrids = create_subgrids(OW_noneddies, grid_size[0], grid_size[1])
        noneddy_grids = []
        for grid in subgrids:
            if not np.ma.is_masked(grid):
                noneddy_grids.append(grid) # Viable non-eddy grid
        # pick same amount of noneddies as eddies randomly from viable non-eddies
        noneddy_grids = random.sample(noneddy_grids, savedImgCounter)
        print(noneddy_grids)
        exit()
        '''
        # Create standard size of grid for which to find random cell
        w = len(lon[0])
        h = len(lat[0])
        gridArea = w*h/nGrids # Area of each grid box
        gridSide = sqrt(gridArea) # Side length of each grid box
        n_lon = floor(w/boxSide)  # Number of boxes that fit along width
        n_lat = floor(h/boxSide)

        # Create nEddies*10 possible subgrids
        subgrids = create_subgrids(data, n_lon, n_lat)
        # Collect all subgrids that does not contain 
        for grid in subgrids:
            rand_lon = np.random.randint(0,n_lon)
            rand_lat = np.random.randint(0,n_lat)


            #rand_image = grid[rand_lon:eddy_shape[0]][rand_lat:eddy_shape[1]]
            #print(rand_image.count)
        '''


    sst_train = np.array(sst_train)
    ssl_train = np.array(ssl_train)
    uvel_train = np.array(uvel_train)
    vvel_train = np.array(vvel_train)
    phase_train = np.array(phase_train)

    # number of "training eddies" :D:D
    nTeddies = sst_train.shape[0]

    # =========================================================
    # ============== Interpolate to largest rect ==============
    # =========================================================

    # The standard grid size we will use
    grid_size = find_max_dim(sst_train[0])

    # Interpolate the images to fit the standard rectangle size. Arrays needs to be float32 numpy arrays for cv2 to do its magic
    # [i] eddie [0] training data ([1] is label)
    for i in range(nTeddies):
        sst_train[i][0] = np.array(sst_train[i][0], dtype='float32') # convert to numpy array
        sst_train[i][0] = cv2.resize(sst_train[i][0], dsize=(frame_size[0], frame_size[1]), interpolation=cv2.INTER_CUBIC) # Resize to a standard size
        ssl_train[i][0] = np.array(ssl_train[i][0], dtype='float32') # convert to numpy array
        ssl_train[i][0] = cv2.resize(ssl_train[i][0], dsize=(frame_size[0], frame_size[1]), interpolation=cv2.INTER_CUBIC) # Resize to a standard size
        uvel_train[i][0] = np.array(uvel_train[i][0], dtype='float32') # convert to numpy array
        uvel_train[i][0] = cv2.resize(uvel_train[i][0], dsize=(frame_size[0], frame_size[1]), interpolation=cv2.INTER_CUBIC) # Resize to a standard size
        vvel_train[i][0] = np.array(vvel_train[i][0], dtype='float32') # convert to numpy array
        vvel_train[i][0] = cv2.resize(vvel_train[i][0], dsize=(frame_size[0], frame_size[1]), interpolation=cv2.INTER_CUBIC) # Resize to a standard size
        phase_train[i][0] = np.array(phase_train[i][0], dtype='float32') # convert to numpy array
        phase_train[i][0] = cv2.resize(phase_train[i][0], dsize=(frame_size[0], frame_size[1]), interpolation=cv2.INTER_CUBIC) # Resize to a standard size


    '''
    # Or do it more complex, but compressed like this? I prefer readable?
    sst_train = np.array([cv2.resize(np.array(sst_train[i]), dsize=(frame_size[0], frame_size[0]), interpolation=cv2.INTER_CUBIC) for i in range(nTeddies)])
    ssl_train = np.array([cv2.resize(np.array(ssl_train[i], dtype='float32'), dsize=(frame_size[1], frame_size[1]), interpolation=cv2.INTER_CUBIC) for i in range(nTeddies)])
    uvel_train = np.array([cv2.resize(np.array(sst_train[i], dtype='float32'), dsize=(frame_size[2], frame_size[2]), interpolation=cv2.INTER_CUBIC) for i in range(nTeddies)])
    vvel_train = np.array([cv2.resize(np.array(ssl_train[i], dtype='float32'), dsize=(frame_size[3], frame_size[3]), interpolation=cv2.INTER_CUBIC) for i in range(nTeddies)])
    phase_train = np.array([cv2.resize(np.array(sst_train[i], dtype='float32'), dsize=(frame_size[4], frame_size[4]), interpolation=cv2.INTER_CUBIC) for i in range(nTeddies)])
    '''


    '''
    # =========================================================
    # ============== Find non-eddy training data ==============
    # =========================================================

    # The arrays looks like this so far: [grid][x/y][lon][lat]

    # Mark anywhere with positive OW value
    OW_noneddies = np.zeros(OW.shape,dtype=int)
    OW_noneddies[np.where(OW > 0)] = 1

    # Find all possible noneddies, then find all viable options and find randomly select n images for each day. We only need to find for one

    # Number of viable noneddy grids we will find for each day
    nGridsPerDay = int(nEddies/len(day))

    # Create standard size of grid for which to find random cell
    w = len(lon[0])
    h = len(lat[0])
    gridArea = w*h/nGrids # Area of each grid box
    gridSide = sqrt(gridArea) # Side length of each grid box
    n_lon = floor(w/boxSide)  # Number of boxes that fit along width
    n_lat = floor(h/boxSide)

    #datasets = [sst, ssl, uvel, vvel, phase]
    #for ds in datasets:
    for day, data in enumerate(ds):
        subgrids = create_subgrids(data, n_lon, n_lat)
        # Collect all subgrids that does not contain 
        for grid in subgrids:
            rand_lon = np.random.randint(0,n_lon)
            rand_lat = np.random.randint(0,n_lat)
            rand_image = grid[rand_lon:eddy_shape[0]][rand_lat:eddy_shape[1]]
            print(rand_image.count)
    '''
    # =========================================================
    # ========= Store as compressed numpy array (npz) =========
    # =========================================================

    print(f"--- Compressing training data")

    # Save data as compressed numpy array
    savez_compressed(f'{args.savedir}/sst_train.npz', sst_train)
    savez_compressed(f'{args.savedir}/ssl_train.npz', ssl_train)
    savez_compressed(f'{args.savedir}/uvel_train.npz', uvel_train)
    savez_compressed(f'{args.savedir}/vvel_train.npz', vvel_train)
    savez_compressed(f'{args.savedir}/phase_train.npz', phase_train)

    print(f"--- Training data complete")


def create_subgrids(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size
    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))



if __name__ == '__main__':
    main()



'''
#"quickscript" to plot and investigate images
fig, axs = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey=True)
lo = lon[lonIdxs]
la = lat[latIdxs]

lo = range(frame_size[0])
la = range(frame_size[1])
axs[0,0].contourf(lo, la, sst_train[0][0], 10, cmap='rainbow')
axs[0,1].contourf(lo, la, ssl_train[0][0], 10, cmap='rainbow')
axs[0,2].contourf(lo, la, uvel_train[0][0], 10, cmap='rainbow')
axs[1,0].contourf(lo, la, vvel_train[0][0], 10, cmap='rainbow')
axs[1,1].contourf(lo, la, phase_train[0][0], 10, cmap='CMRmap')
title = dateStr + "_" + check_cyclone(cyclone)
fig.suptitle(title, fontsize=16)

guiEvent, guiValues = show_figure(fig)
plt.close(fig)
'''