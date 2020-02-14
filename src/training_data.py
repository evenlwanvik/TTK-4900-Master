from training_data.eddies import load_netcdf4,eddy_detection,dataframe_eddies,plot_eddies,julianh2gregorian
from matplotlib.patches import Rectangle
from numpy import savez_compressed
import matplotlib.pyplot as plt
from training_data.gui import show_figure
from tools.bfs import bfs
from datetime import date
from math import cos, pi
from operator import eq
import tools.dim as dim
import xarray as xr
import numpy as np
import itertools
import argparse
import datetime
import logging
import random
import cv2
import os
import io
import sys

from noneddy import get_noneddies

argp = argparse.ArgumentParser()
argp.add_argument("-fp", "--fpath", default='C:/Master/data/cmems_data/global_10km/phys_noland_001.nc', help="rectangular patche size multiplier")
argp.add_argument("-rs", "--size", default=1.5, help="rectangular patche size multiplier")
argp.add_argument("-sd", "--savedir", default='C:/Master/TTK-4900-Master/data/training_data/200_days_2018', help="training data save dir")


logPath = f"{os.path.dirname(os.path.realpath(__file__))}/log"
logName = f"{datetime.datetime.now().strftime('%d%m%Y_%H%M')}.log"

if not os.path.exists(logPath):
    os.makedirs(logPath)

# create logger 
logger = logging.getLogger("Training Data")
logger.setLevel(logging.INFO)
# create file handler 
fh = logging.FileHandler("{0}/{1}".format(logPath, logName))
fh.setLevel(logging.INFO)
# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

def lon2km(lon, lat):
    """ Convert from longitudinal displacement to km """
    return lon * 111.320e3 * cos(lat)
    

def lat2km(lat):
    """ Convert from latitudinal displacement to km """
    return 110.54e3 * lat


def index_list(ncols,nrows):
    """" Create an array of dimension nrows x ncols with indeces as values """
    return [[(i,j) for j in range(nrows)] for i in range(ncols)]


def random_grids(arr, nOut):
    """ Get nOut random grids from arr """
    nTot = dim.shape(arr)[0]
    x = random.sample(range(nTot), nOut)
    return [ arr[i] for i in x ]


def create_subgrids(arr, nrows, ncols, inner=1):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size
    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    'inner' tells the dimension of the array elements, i.e. 2 if tuple, 1 if single element
    """
    h, w = dim.shape(arr)[0:2]
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.redim.shape(h//nrows, nrows, -1, ncols, inner)# last 2 is because array consists of 2d idx
               .swapaxes(1,2)
               .redim.shape(-1, nrows, ncols, inner)) 


def plot_grids(data, lon, lat, title="__"):
    #"quickscript" to plot and investigate images
    fig, axs = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey=True)
    axs[0,0].contourf(lon, lat, data[0].T, 10, cmap='rainbow')
    axs[0,1].contourf(lon, lat, data[1].T, 10, cmap='rainbow')
    axs[0,2].contourf(lon, lat, data[2].T, 10, cmap='rainbow')
    axs[1,0].contourf(lon, lat, data[3].T, 10, cmap='rainbow')
    axs[1,1].contourf(lon, lat, data[4].T, 10, cmap='CMRmap')
    n=-1
    color_array = np.sqrt(((data[2]-n)/2)**2 + ((data[3]-n)/2)**2)
    axs[1,2].quiver(lon, lat, data[2].T, data[3].T, color_array, scale=7) # Plot vector field
    #axs[1,2].contourf(lon, lat, data[5].T, 10) # Or plot the OW values
    
    fig.suptitle(title, fontsize=16)

    guiEvent, guiValues = show_figure(fig)
    plt.close(fig)

    return guiEvent


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

    logger.info("loading netcdf")

    # load data
    (ds,t,lon,lat,depth,uvel_full,vvel_full,sst_full,ssl_full) =  load_netcdf4(args.fpath)

    # Confidence level, usually 90%
    R2_criterion = 0.95

    # OW value at which to begin the evaluation of R2, default was -1, want to use -8 to be absolutely sure
    OW_start = -8.0

    # Number of local minima to evaluate using R2 method.
    # Set low (like 20) to see a few R2 eddies quickly.
    # Set high (like 1e5) to find all eddies in domain.
    max_evaluation_points = 10000 

    # Minimum number of cells required to be identified as an eddie.
    min_eddie_cells = 4 # set to 3 to be coherent with the use of the R2 method, 3 points seems like a reasonable minimun for a correlation 

    # z-level to plot.  Usually set to 0 for the surface.
    k_plot = 0

    # Create eddy images for each day in datase
    #for day, time in enumerate(t):
    # Shuffle the time so that the expert won't see the same long-lasting eddies
    for day in random.sample(range(0, len(t)), len(t)): 

        dateStr = "{:%d-%m-%Y}".format(datetime.date(1950, 1, 1) + datetime.timedelta(hours=float(t[day])) )
        logger.info(f"Creating images for dataset {dateStr}")

        # create a text trap
        text_trap = io.StringIO()
        sys.stdout = text_trap

        # Run the OW-R2 algorithm
        lon,lat,u,v,vorticity,OW,OW_eddies,eddie_census,nEddies,circulation_mask = eddy_detection(
                lon,lat,depth,uvel_full,vvel_full,day,R2_criterion,OW_start,max_evaluation_points,min_eddie_cells)

        # restore stdout
        sys.stdout = sys.__stdout__

        sst_train = []
        ssl_train = []
        uvel_train = []
        vvel_train = []
        phase_train = []
        nDataset = 5

        # =========================================================
        # ============== Prepare datasets and lists ===============
        # =========================================================

        eddyCtrIdx = []
        for i in range(0,nEddies):
            lonIdx = np.argmax(lon>eddie_census[2,i])-1
            latIdx = np.argmax(lat>eddie_census[3,i])-1
            eddyCtrIdx.append( (lonIdx, latIdx) )

        # Netcdf uses (lat,lon) we want to use (lon,lat) and discard the depth
        sst = sst_full[day,:,:].T
        ssl = ssl_full[day,:,:].T
        uvel = uvel_full[day,0,:,:].T
        vvel = vvel_full[day,0,:,:].T
        # Calculate the phase angle (direction) of the current
        with np.errstate(all='ignore'): # Disable zero div warning
            phase = xr.ufuncs.rad2deg( xr.ufuncs.arctan2(vvel, uvel) ) + 180
        OW = OW[:,:,0]
        nLon = len(lon)
        nLat = len(lat)

        datasets = (sst, ssl, uvel, vvel, phase, OW) 
        
        # =========================================================
        # ======= Create rectangular patches around eddies ========
        # =========================================================

        logger.info(f"+ Creating rectangles for {nEddies} eddies")

        savedImgCounter = 0 # saved image counter for file ID
        for eddyId, ctrIdx in enumerate(eddyCtrIdx): # nEddies

            ctrCoord = lon[ctrIdx[0]], lat[ctrIdx[1]]
            diameter_km = eddie_census[5][eddyId]

            bfs_diameter_km, bfs_center = eddy_metrics(OW_eddies, ctrIdx, lon, lat)

            # Positive rotation (counter-clockwise) is a cyclone in the northern hemisphere because of the coriolis effect
            if (eddie_census[1][eddyId] > 0.0): cyclone = 1 # 1 is a cyclone, 0 is nothing and -1 is anti-cyclone (negative rotation)
            else: cyclone = -1

            logger.info(f"+++ Creating rectangles for {check_cyclone(cyclone)} with center {ctrCoord} and diameter {diameter_km}")
            
            # Find rectangle metrics
            height = args.size * abs(diameter_km / 110.54) # 1 deg = 110.54 km, 1.2 to be sure the image covers the eddy
            width = args.size * abs(diameter_km / (111.320 * cos(lat[ctrIdx[1]]))) # 1 deg = 111.320*cos(latitude) km, using center latitude as ref
            
            #-TODO: I DO THIS TWICE NOW! CREATE FUNCTION? -----

            lon_bnds = ctrCoord[0]-width/2.0, ctrCoord[0]+width/2.0
            lat_bnds = ctrCoord[1]-height/2.0, ctrCoord[1]+height/2.0
            
            # Indeces of current eddy image
            lonIdxs = np.where((lon > lon_bnds[0]) & (lon < lon_bnds[1]))[0]
            latIdxs = np.where((lat > lat_bnds[0]) & (lat < lat_bnds[1]))[0]

            eddy_data = np.array([np.zeros((lonIdxs.size,latIdxs.size)) for _ in range(6)])
            
            for i, lo in enumerate(lonIdxs):
                for j, la in enumerate(latIdxs):
                    for k, eddy in enumerate(datasets):
                        eddy_data[k,i,j] = eddy[lo,la]

            lo = lon[lonIdxs]
            la = lat[latIdxs]
            title = dateStr + "_" + check_cyclone(cyclone)
  
            # Plot and flag to save eddy
            #add = plot_grids(eddy_data, lo, la, title)

            #----------------------------------------------------------

            # Find the center from water level
            if cyclone==1:
                idx = np.unravel_index(eddy_data[1].argmax(), eddy_data[1].shape)
                ctrCoord = lon[lonIdxs[idx[0]]], lat[latIdxs[idx[1]]]
                logger.info(f"+++ Argmax center -> lon: {ctrCoord[0]}, Center lat: {ctrCoord[1]}")
            else:
                idx = np.unravel_index(eddy_data[1].argmin(), eddy_data[1].shape)
                ctrCoord = lon[lonIdxs[idx[0]]], lat[latIdxs[idx[1]]]
                logger.info(f"+++ Argmin center -> lon: {ctrCoord[0]}, Center lat: {ctrCoord[1]}")

             #-TODO: I DO IT AGAIN TO FIND WATER LEVEL CENTER -----

            lon_bnds = ctrCoord[0]-width/2.0, ctrCoord[0]+width/2.0
            lat_bnds = ctrCoord[1]-height/2.0, ctrCoord[1]+height/2.0
            
            # Indeces of current eddy image
            lonIdxs = np.where((lon > lon_bnds[0]) & (lon < lon_bnds[1]))[0]
            latIdxs = np.where((lat > lat_bnds[0]) & (lat < lat_bnds[1]))[0]

            eddy_data = np.array([np.zeros((lonIdxs.size,latIdxs.size)) for _ in range(6)])
            
            for i, lo in enumerate(lonIdxs):
                for j, la in enumerate(latIdxs):
                    for k, eddy in enumerate(datasets):
                        eddy_data[k,i,j] = eddy[lo,la]

            #----------------------------------------------------------
            
            lo = lon[lonIdxs]
            la = lat[latIdxs]
            title = dateStr + "_" + check_cyclone(cyclone)
  
            # Plot and flag to save eddy
            add = plot_grids(eddy_data, lo, la, title)

            #guiEvent, guiValues = show_figure(fig)
            #add = 'Yes' # Bypass GUI selection
            if add=='Yes':
                savedImgCounter = savedImgCounter + 1
                # Create images?
                '''
                dirPath = 'C:/Master/TTK-4900-Master/images/'+dateStr+'/'
                if not os.path.exists(dirPath):
                    os.makedirs(dirPath)
                imPath = dirPath + title + f"_{savedImgCounter}.png"   
                plt.savefig(imPath, bbox_inches='tight')
                '''

                sst_train.append([eddy_data[0], cyclone]) # [data, label]
                ssl_train.append([eddy_data[1], cyclone]) 
                uvel_train.append([eddy_data[2], cyclone]) 
                vvel_train.append([eddy_data[3], cyclone]) 
                phase_train.append([eddy_data[4], cyclone]) 

                logger.info(f"+++++ Saving image {eddyId} as an eddy")   

            else: 
                logger.info(f"+++++ Discarding image {eddyId}")
            
        # =========================================================
        # ================ Select non-eddy images =================
        # =========================================================

        if savedImgCounter <= 0:
            logger.info(f"+++++ No eddies found")
            continue   

        # Subgrid (sg) longitude and latitude length
        sgLon, sgLat = dim.find_avg_dim(sst_train) 

        loRange, laRange = range(0, nLon, sgLon), range(0, nLat, sgLat)
    
        # Create OW array of compatible dimensions for comparing masks
        OW_noeddy = OW[:loRange[-1],:laRange[-1]]
        OW_noeddy = create_subgrids( np.ma.masked_where(OW_noeddy < -0.8, OW_noeddy), sgLon, sgLat, 1 )

        # Get a 2d grid of indeces -> make it moldable to the average grid -> convert to subgrids
        idx_subgrids = create_subgrids( np.array( index_list(nLon, nLat) )[:loRange[-1],:laRange[-1]], sgLon, sgLat, 2 )

        noneddy_idx_subgrids = []
        for i, grid in enumerate(OW_noeddy):
            if not np.ma.is_masked(grid):
                noneddy_idx_subgrids.append(idx_subgrids[i])

        nNoneddies = len(noneddy_idx_subgrids)
        data_noeddy = np.array([[np.zeros((sgLon,sgLat)) for _ in range(nNoneddies)] for _ in range(6)])
        
        # Shuffle the noneddies and loop thorugh untill we have chosen the same amount of non-eddies as eddies
        random.shuffle(noneddy_idx_subgrids)
        added = 0
        for grid_id, idx_grid in enumerate(noneddy_idx_subgrids):
            OW_ = np.zeros((idx_grid.shape[:2]))
            for i in range(len(idx_grid)):
                for j in range(len(idx_grid[0])):
                    idx = idx_grid[i,j][0], idx_grid[i,j][1]
                    for k in range(len(data_noeddy)):
                        data_noeddy[k,grid_id,i,j] = datasets[k][idx]
            #print(idx_grid)
            lo, la = lon[idx_grid[:,0,0]], lat[idx_grid[0,:,1]]
            title = dateStr + "_noeddy"
            add = plot_grids(data_noeddy[:,grid_id,:,:], lo, la, title)
            if add=='Yes':
                added = added + 1
                sst_train.append([data_noeddy[0,grid_id,:,:], 0]) # [data, label]
                ssl_train.append([data_noeddy[0,grid_id,:,:], 0]) 
                uvel_train.append([data_noeddy[0,grid_id,:,:], 0]) 
                vvel_train.append([data_noeddy[0,grid_id,:,:], 0]) 
                phase_train.append([data_noeddy[0,grid_id,:,:], 0])
                logger.info(f"+++++ Saving noneddy")       
            if added >= savedImgCounter:
                break

        # =========================================================
        # ============== Interpolate ==============
        # =========================================================

        #sst_out = np.array(sst_train)
        #ssl_out = np.array(ssl_train)
        #uvel_out = np.array(uvel_train)
        #vvel_out = np.array(vvel_train)
        #phase_out = np.array(phase_train)
        #nTeddies = sst_out.shape[0]


        logger.info(f"Compressing and storing training data so far")


        # =========================================================
        # ========== Save data as compressed numpy array ==========
        # =========================================================

        # If folder doesn't exist, create folder and just save the data for the first day
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)
            savez_compressed( f'{args.savedir}/sst_train.npz', sst_train)
            savez_compressed( f'{args.savedir}/ssl_train.npz', ssl_train)
            savez_compressed( f'{args.savedir}/uvel_train.npz', uvel_train)
            savez_compressed( f'{args.savedir}/vvel_train.npz', vvel_train)
            savez_compressed( f'{args.savedir}/phase_train.npz', phase_train)
        # If not, we open and append to the existing data
        else:
            with np.load(f'{args.savedir}/sst_train.npz', 'w+', allow_pickle=True) as data:
                savez_compressed( f'{args.savedir}/sst_train.npz', np.append(data['arr_0'], sst_train, axis=0))
            with np.load(f'{args.savedir}/ssl_train.npz', 'w+', allow_pickle=True) as data:
                savez_compressed( f'{args.savedir}/ssl_train.npz', np.append(data['arr_0'], ssl_train, axis=0))
            with np.load(f'{args.savedir}/uvel_train.npz', 'w+', allow_pickle=True) as data:
                savez_compressed(f'{args.savedir}/uvel_train.npz', np.append(data['arr_0'], uvel_train, axis=0))
            with np.load(f'{args.savedir}/vvel_train.npz', 'w+', allow_pickle=True) as data:
                savez_compressed(f'{args.savedir}/vvel_train.npz', np.append(data['arr_0'], vvel_train, axis=0))
            with np.load(f'{args.savedir}/phase_train.npz', 'w+', allow_pickle=True) as data:
                savez_compressed(f'{args.savedir}/phase_train.npz', np.append(data['arr_0'], phase_train, axis=0))

    '''
    # number of training eddies -- Allow me to introduce (drumroll) "Teddies!" :D:D
    nTeddies = sst_train.shape[0]

    # =========================================================
    # ============== Interpolate to largest rect ==============
    # =========================================================

    # The standard grid size we will use
    grid_size = dim.find_avg_dim(sst_train[0])

    # Interpolate the images to fit the standard rectangle size. Arrays needs to be float32 numpy arrays for cv2 to do its magic
    # [i] eddie [0] training data ([1] is label)
    for i in range(nTeddies):
        sst_train[i][0] = np.array(sst_train[i][0], dtype='float32') # convert to numpy array
        sst_train[i][0] = cv2.resize(sst_train[i][0], dsize=(grid_size[0], grid_size[1]), interpolation=cv2.INTER_CUBIC) # Resize to a standard size
        ssl_train[i][0] = np.array(ssl_train[i][0], dtype='float32') # convert to numpy array
        ssl_train[i][0] = cv2.resize(ssl_train[i][0], dsize=(grid_size[0], grid_size[1]), interpolation=cv2.INTER_CUBIC) # Resize to a standard size
        uvel_train[i][0] = np.array(uvel_train[i][0], dtype='float32') # convert to numpy array
        uvel_train[i][0] = cv2.resize(uvel_train[i][0], dsize=(grid_size[0], grid_size[1]), interpolation=cv2.INTER_CUBIC) # Resize to a standard size
        vvel_train[i][0] = np.array(vvel_train[i][0], dtype='float32') # convert to numpy array
        vvel_train[i][0] = cv2.resize(vvel_train[i][0], dsize=(grid_size[0], grid_size[1]), interpolation=cv2.INTER_CUBIC) # Resize to a standard size
        phase_train[i][0] = np.array(phase_train[i][0], dtype='float32') # convert to numpy array
        phase_train[i][0] = cv2.resize(phase_train[i][0], dsize=(grid_size[0], grid_size[1]), interpolation=cv2.INTER_CUBIC) # Resize to a standard size

    # =========================================================
    # ========= Store as compressed numpy array (npz) =========
    # =========================================================

    print(f"Compressing training data")

    # Save data as compressed numpy array
    savez_compressed(f'{args.savedir}/sst_train.npz', sst_train)
    savez_compressed(f'{args.savedir}/ssl_train.npz', ssl_train)
    savez_compressed(f'{args.savedir}/uvel_train.npz', uvel_train)
    savez_compressed(f'{args.savedir}/vvel_train.npz', vvel_train)
    savez_compressed(f'{args.savedir}/phase_train.npz', phase_train)

    print(f"Training data complete")
    '''

if __name__ == '__main__':
    main()
