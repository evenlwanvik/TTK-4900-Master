from training_data.eddies import eddy_detection,dataframe_eddies,plot_eddies,julianh2gregorian
from tools.machine_learning import sliding_window
from matplotlib.patches import Rectangle
from tools.load_nc import load_netcdf4
from numpy import savez_compressed
import matplotlib.pyplot as plt
from tools.bfs import bfs
from datetime import date
from math import cos, pi
from operator import eq
import tools.dim as dim
from tools import gui
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

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

argp = argparse.ArgumentParser()
argp.add_argument("-fp", "--fpath", default='C:/Master/data/cmems_data/global_10km/phys_noland_2016_001.nc', help="rectangular patch size multiplier")
argp.add_argument("-rs", "--size", default=1.3, help="rectangular patche size multiplier")
argp.add_argument("-sd", "--savedir", default='C:/Master/TTK-4900-Master/data/training_data/200_days_2016', help="training data save dir")


logPath = f"{os.path.dirname(os.path.realpath(__file__))}/training_data/log"
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
    arr = np.array(arr)
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols, inner)# last 2 is because array consists of 2d idx
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols, inner)) 


def plot_grids(data, lon, lat, larger_grid=None, title="__"):
    #"quickscript" to plot and investigate images
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # levels for the phase angle to make it not interpolate 
    levels = MaxNLocator(nbins=10).tick_values(data[4].min(), data[4].max())
    cmap = plt.get_cmap('CMRmap')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    axs[0,0].pcolormesh(lon, lat, data[4].T, cmap=cmap, norm=norm)

    axs[0,1].contourf(lon, lat, data[1].T, 20, cmap='rainbow')
    n=-1
    color_array = np.sqrt(((data[2]-n)/2)**2 + ((data[3]-n)/2)**2)
    axs[1,0].quiver(lon, lat, data[2].T, data[3].T, color_array, scale=5) # Plot vector field
    if larger_grid is not None:
        axs[1,1].contourf(larger_grid[0], larger_grid[1], larger_grid[2].T, 20, cmap='rainbow') # show a larger parcel to analyze the surroundings
    #axs[1,2].contourf(lon, lat, data[5].T, 10) # Or plot the OW values
    
    fig.suptitle(title, fontsize=16)

    guiEvent, guiValues = gui.show_figure(fig)
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
    # positive 1 denotes positive rotation (counter-clockwise), which is a cyclone in the norther hemisphere
    if   flag==1:  return "cyclone"
    elif flag==-1: return "anticyclone"
    else:          return "nothing"


def save_npz_array(data):
    # If folder doesn't exist, create folder and just save the data for the first day
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
        savez_compressed( f'{args.savedir}/sst_train.npz', data[0])
        savez_compressed( f'{args.savedir}/ssl_train.npz', data[1])
        savez_compressed( f'{args.savedir}/uvel_train.npz', data[2])
        savez_compressed( f'{args.savedir}/vvel_train.npz', data[3])
        savez_compressed( f'{args.savedir}/phase_train.npz', data[4])
    # If not, we open and append to the existing data
    else:
        with np.load(f'{args.savedir}/sst_train.npz', 'w+', allow_pickle=True) as data:
            savez_compressed( f'{args.savedir}/sst_train.npz', np.append(data['arr_0'], data[0], axis=0))
        with np.load(f'{args.savedir}/ssl_train.npz', 'w+', allow_pickle=True) as data:
            savez_compressed( f'{args.savedir}/ssl_train.npz', np.append(data['arr_0'], data[1], axis=0))
        with np.load(f'{args.savedir}/uvel_train.npz', 'w+', allow_pickle=True) as data:
            savez_compressed(f'{args.savedir}/uvel_train.npz', np.append(data['arr_0'], data[2], axis=0))
        with np.load(f'{args.savedir}/vvel_train.npz', 'w+', allow_pickle=True) as data:
            savez_compressed(f'{args.savedir}/vvel_train.npz', np.append(data['arr_0'], data[3], axis=0))
        with np.load(f'{args.savedir}/phase_train.npz', 'w+', allow_pickle=True) as data:
            savez_compressed(f'{args.savedir}/phase_train.npz', np.append(data['arr_0'], data[4], axis=0))


def semi_automatic_training():

    args, leftovers = argp.parse_known_args()

    logger.info("loading netcdf")

    # load data
    (ds,t,lon,lat,depth,uvel_full,vvel_full,sst_full,ssl_full) =  load_netcdf4(args.fpath)

    # Confidence level, usually 90%
    R2_criterion = 0.90

    # OW value at which to begin the evaluation of R2, default was -1, want to use -8 to be absolutely sure
    OW_start = -6.0

    # Number of local minima to evaluate using R2 method.
    # Set low (like 20) to see a few R2 eddies quickly.
    # Set high (like 1e5) to find all eddies in domain.
    max_evaluation_points = 100000 

    # Minimum number of cells required to be identified as an eddie.
    min_eddie_cells = 3 # set to 3 to be coherent with the use of the R2 method, 3 points seems like a reasonable minimun for a correlation 

    # z-level to plot.  Usually set to 0 for the surface.
    k_plot = 0

    dlon = abs(lon[0]-lon[1])
    dlat = abs(lat[0]-lat[1])

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

            lon_bnds = ctrCoord[0]-width/2.0, ctrCoord[0]+width/2.0
            lat_bnds = ctrCoord[1]-height/2.0, ctrCoord[1]+height/2.0
            
            # Indeces of current eddy image
            lonIdxs = np.where((lon >= lon_bnds[0]) & (lon <= lon_bnds[1]))[0]
            latIdxs = np.where((lat >= lat_bnds[0]) & (lat <= lat_bnds[1]))[0]

            eddy_data = np.array([np.zeros((lonIdxs.size,latIdxs.size)) for _ in range(6)])
  
            # Plot and flag to save eddy
            #add = plot_grids(eddy_data, lo, la, title)

            #-------- Move closer to center of eddy ------------

            title = dateStr + "_" + check_cyclone(cyclone)

            choices = ('Center', 'incLon', 'incLat', 'decLon', 'decLat')
            response = 'Center'
            #response = 'Yes' # Skip this section for debugging non-eddy section
            while response in choices:

                lo = lon[lonIdxs]
                la = lat[latIdxs]

                for i, loIdx in enumerate(lonIdxs):
                    for j, laIdx in enumerate(latIdxs):
                        for k, measurement in enumerate(datasets): # for every measurement type in datasets
                            eddy_data[k,i,j] = measurement[loIdx,laIdx]

                # Store a larger grid to make it easier to see if we have an eddy and if we should center image 
                if (lonIdxs[0]-5 < 0 or lonIdxs[-1]+5 >= nLon) or (latIdxs[0]-3 < 0 or latIdxs[-1]+3 >= nLat):
                    larger_grid = None
                else:
                    larger_grid = [ np.zeros(lonIdxs.size+10), np.zeros(latIdxs.size+6), 
                                    np.zeros((lonIdxs.size+10,latIdxs.size+6)), ]
                    for i, loIdx in enumerate(range(lonIdxs[0]-5, lonIdxs[-1]+6)):
                        for j, laIdx in enumerate(range(latIdxs[0]-3, latIdxs[-1]+4)):
                            larger_grid[0][i] = lon[loIdx]
                            larger_grid[1][j] = lat[laIdx]
                            larger_grid[2][i,j] = ssl[loIdx,laIdx]

                response = plot_grids(eddy_data, lo, la, larger_grid, title)
                if response not in choices: # TODO: feel like this is a silly way of doing this
                    break
                if response == 'Center':
                    # Find the center from water level
                    logger.info(f"+++ Centering eddy towards a minima/maxima depending on eddy type")
                    if cyclone==1:
                        idx = np.unravel_index(eddy_data[1].argmax(), eddy_data[1].shape)
                        ctrCoord = lon[lonIdxs[idx[0]]], lat[latIdxs[idx[1]]]
                        logger.info(f"+++ Argmax center -> lon: {ctrCoord[0]}, Center lat: {ctrCoord[1]}")
                    else:
                        idx = np.unravel_index(eddy_data[1].argmin(), eddy_data[1].shape)
                        ctrCoord = lon[lonIdxs[idx[0]]], lat[latIdxs[idx[1]]]
                        logger.info(f"+++ Argmin center -> lon: {ctrCoord[0]}, Center lat: {ctrCoord[1]}")

                    # New width and height in case we've moved in lon/lat direction
                    width, height = abs(lo[0]-lo[-1])+dlon, abs(la[0]-la[-1])+dlat

                    lon_bnds = ctrCoord[0]-width/2.0, ctrCoord[0]+width/2.0
                    lat_bnds = ctrCoord[1]-height/2.0, ctrCoord[1]+height/2.0

                    # Indeces of current eddy image
                    lonIdxs = np.where((lon >= lon_bnds[0]) & (lon <= lon_bnds[1]))[0]
                    latIdxs = np.where((lat >= lat_bnds[0]) & (lat <= lat_bnds[1]))[0]

                elif response == 'incLon':
                    if (lonIdxs[0] <= 0 or lonIdxs[-1] >= nLon): 
                        logger.info(f"+++ Longitude can't be increased further")
                    else:
                        lonIdxs = np.arange(lonIdxs[0]-1, lonIdxs[-1]+2)
                        logger.info(f"+++ Increasing lontitude by 1 cell in both directions to ({lonIdxs[0]}:{lonIdxs[-1]})")
                elif response == 'incLat':
                    if (latIdxs[0] <= 0 or latIdxs[-1] >= nLat): 
                        logger.info(f"+++ Latitude can't be increased further")
                    else:
                        latIdxs = np.arange(latIdxs[0]-1, latIdxs[-1]+2)
                        logger.info(f"+++ Increasing latitude by 1 cell in both directions to ({latIdxs[0]}:{latIdxs[-1]})")
                elif response == 'decLon':
                    lonIdxs = np.arange(lonIdxs[0]+1, lonIdxs[-1])
                    logger.info(f"+++ Decreasing lontitude by 1 cell in both directions to ({lonIdxs[0]}:{lonIdxs[-1]})")
                elif response == 'decLat':
                    latIdxs = np.arange(latIdxs[0]+1, latIdxs[-1])
                    logger.info(f"+++ Decreasing latitude by 1 cell in both directions to ({latIdxs[0]}:{latIdxs[-1]})")
                eddy_data = np.array([np.zeros((lonIdxs.size,latIdxs.size)) for _ in range(6)])      

            #----------------------------------------------------------
            
            lo = lon[lonIdxs]
            la = lat[latIdxs]

            #guiEvent, guiValues = show_figure(fig)
            #add = 'Yes' # Bypass GUI selection
            if response=='Yes':
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
        sgLon, sgLat = dim.find_avg_dim(sst_train, start_axis=0) 
        logger.info(f"+++++ Using average dimensions ({sgLon}, {sgLat}) for non-eddy")

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
            title = dateStr + "_noneddy"
            add = plot_grids(data_noeddy[:,grid_id,:,:], lo, la, None, title)
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

        #save_npz_array( (sst_train, ssl_train, uvel_train, vvel_train, phase_train) )

def adjustment_data():
    ''' Method to run the ML model to provide correctional non-eddy images for the model '''

    (ds,t,lon,lat,depth,uvel_full,vvel_full,sst_full,ssl_full) =  load_netcdf4(args.fpath)

    ssl_probLim = 0.96
    phase_probLim = 0.35
    stepSize = 8
    scaler = MinMaxScaler(feature_range=(0,1))

    ssl_clf   = load_model('models/cnn_multiclass_ssl_01.h5')
    phase_clf = load_model('models/cnn_multiclass_phase_01.h5')

    winW, winH = int(16), int(10)
    dSize = (winW, winH)

    # Lists that will hold the training data
    sst_train = []
    ssl_train = []
    uvel_train = []
    vvel_train = []
    phase_train = []
    nDataset = 5

    # Shuffle the time so that the expert won't see the same long-lasting eddies
    for i, day in enumerate(random.sample(range(0, len(t)), len(t))): 

        ssl = np.array(ssl_full[day].T, dtype='float32') 
        sst = np.array(sst_full[day,0].T, dtype='float32') 
        uvel = np.array(uvel_full[day,0].T, dtype='float32') 
        vvel = np.array(vvel_full[day,0].T, dtype='float32') 
        with np.errstate(all='ignore'): # Disable zero div warning
            phase = xr.ufuncs.rad2deg( xr.ufuncs.arctan2(vvel, uvel) ) + 180

        shape = ssl.shape
        ssl_scaled = scaler.fit_transform(ssl)
        phase_scaled = scaler.fit_transform(phase)

        # loop over the sliding window of indeces
        for x, y, (lonIdxs, latIdxs) in sliding_window(ssl, stepSize=stepSize, windowSize=dSize):

            if lonIdxs[-1] >= shape[0] or latIdxs[-1] >= shape[1]:
                continue
            dSize = (winH, winW)
            # Window indexed data and resizing from a smaller window to model size
            ssl_wind = np.array([[ssl[i,j] for j in latIdxs] for i in lonIdxs])
            ssl_scaled_wind = np.array([[ssl_scaled[i,j] for j in latIdxs] for i in lonIdxs])
            phase_wind = np.array([[phase[i,j] for j in latIdxs] for i in lonIdxs])
            phase_scaled_wind = np.array([[phase_scaled[i,j] for j in latIdxs] for i in lonIdxs])
            uvel_wind = np.array([[uvel[i,j] for j in latIdxs] for i in lonIdxs])
            vvel_wind = np.array([[vvel[i,j] for j in latIdxs] for i in lonIdxs])
            sst_wind = np.array([[sst[i,j] for j in latIdxs] for i in lonIdxs])

            # Add channel dimension for CNN
            ssl_cnn_window   = np.expand_dims(np.expand_dims(ssl_scaled_wind, 2), 0)
            phase_cnn_window = np.expand_dims(np.expand_dims(ssl_scaled_wind, 2), 0)

            lo, la = lon[lonIdxs], lat[latIdxs]

            # Predict and receive probability
            ssl_prob   = ssl_clf.predict_proba(ssl_cnn_window)
            phase_prob = phase_clf.predict_proba(phase_cnn_window)

            # By default we say we have a non-eddy (cyclone flag)
            cyclone_f = 0
            # If second column is larger than the boundary, we have a anti-cyclone
            if ssl_prob[0,1] > ssl_probLim: cyclone_f = -1
            # If third column is larger, we have a cyclone
            elif ssl_prob[0,2] > ssl_probLim: cyclone_f = 1

            eddy_data = [ssl_wind, phase_wind, uvel_wind, vvel_wind]
            
            # Plot and flag if the prediction is correct or not
            yes_no = plot_grids(eddy_data, lo, la, check_cyclone(cyclone_f))
            # Add to training data if expert labels it correct
            if yes_no == 'Yes':
                sst_train.append([sst_wind, cyclone_f]) # [data, label]
                ssl_train.append([ssl_wind, cyclone_f]) 
                uvel_train.append([ssl_wind, cyclone_f]) 
                vvel_train.append([ssl_wind, cyclone_f]) 
                phase_train.append([ssl_wind, cyclone_f])
            # If not, change  the label to non-eddy
            elif yes_no == 'No':
                sst_train.append([sst_wind, 0]) # [data, label]
                ssl_train.append([ssl_wind, 0]) 
                uvel_train.append([ssl_wind, 0]) 
                vvel_train.append([ssl_wind, 0]) 
                phase_train.append([ssl_wind, 0])
            # Every 10 sample add to the compressed array
            if i%10==0:
                # ADD TO THE COMPRESSED NUMPY ARRAY
                save_npz_array( (sst_train, ssl_train, uvel_train, vvel_train, phase_train) )    


if __name__ == '__main__':
    semi_automatic_training()
