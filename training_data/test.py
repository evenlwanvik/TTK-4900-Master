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
import itertools
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

def shape(a):
    """ Get the shape of list """
    return (np.array(a).shape)

def find_max_dim(a):
    """" Find the largest dimension of list or array """
    return max( [dim(i[0]) for i in a])


def find_avg_dim(a):
    """" Find the average frame size for both col and row, [0] since we want the training data, and not the label """
    x = np.array([dim(i[0]) for i in a])
    return x.mean(axis=0, dtype=int)


def index_list(ncols,nrows):
    """" Create an array of dimension nrows x ncols with indeces as values """
    return [[(i,j) for j in range(nrows)] for i in range(ncols)]


def random_grids(arr, nOut):
    """ Get nOut random grids from arr """
    nTot = shape(arr)[0]
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
    h, w = shape(arr)[0:2]
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols, inner)# last 2 is because array consists of 2d idx
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols, inner)) 


def plot_grids(data, idx_grid, lon, lat, OW=None, title="__"):
    #"quickscript" to plot and investigate images
    fig, axs = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey=True)
    lo, la = idx_grid[:,0,0], idx_grid[0,:,1]
    axs[0,0].contourf(lo, la, data[0].T, 10, cmap='rainbow')
    axs[0,1].contourf(lo, la, data[1].T, 10, cmap='rainbow')
    axs[0,2].contourf(lo, la, data[2].T, 10, cmap='rainbow')
    axs[1,0].contourf(lo, la, data[3].T, 10, cmap='rainbow')
    axs[1,1].contourf(lo, la, data[4].T, 10, cmap='CMRmap')
    if OW is not None:
        axs[1,2].contourf(lo, la, OW.T, 10)

    fig.suptitle(title, fontsize=16)

    guiEvent, guiValues = show_figure(fig)
    plt.close(fig)


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

        # =========================================================
        # ============== Prepare datasets and lists ===============
        # =========================================================

        # Fetch all the center indeces from the eddie cencus
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
        OW = OW[:,:,0]

        # Calculate the phase angle (direction) of the current
        with np.errstate(all='ignore'): # Disable zero div warning
            phase = xr.ufuncs.rad2deg( xr.ufuncs.arctan2(vvel, uvel) ) + 180

        # Aggregate data
        datasets = (sst, ssl, uvel, vvel, phase, OW)

        # Dimensions of grid
        nLon = len(lon)
        nLat = len(lat)

        print(f"---++ Creating rectangles for {nEddies} eddies")

        for eddyId, ctrIdx in enumerate(eddyCtrIdx): # nEddies

            ctrCoord = lon[ctrIdx[0]], lat[ctrIdx[1]]
            diameter_km = eddie_census[5][eddyId]
            
            # Use BFS to find centre and diameter?
            #bfs_diameter_km, bfs_center = eddy_metrics(OW_eddies, ctrIdx, lon, lat)

            print(f"---++++ Creating rectangles for eddy {eddyId} with center {ctrCoord} and diameter {diameter_km}")

            # Find rectangle metrics
            height = args.size * abs(diameter_km / 110.54) # 1 deg = 110.54 km, 1.2 to be sure the image covers the eddy
            width = args.size * abs(diameter_km / (111.320 * cos(lat[ctrIdx[1]]))) # 1 deg = 111.320*cos(latitude) km, using center latitude as ref

            lon_bnds = ctrCoord[0]-width/2.0, ctrCoord[0]+width/2.0
            lat_bnds = ctrCoord[1]-height/2.0, ctrCoord[1]+height/2.0

            # Indeces of current eddy image
            lonIdxs = np.where((lon > lon_bnds[0]) & (lon < lon_bnds[1]))[0]
            latIdxs = np.where((lat > lat_bnds[0]) & (lat < lat_bnds[1]))[0]

            sst_eddy = [np.zeros((lonIdxs.size,latIdxs.size))]


            data_noeddy = np.array([[np.zeros((sgLon,sgLat)) for _ in range(nNoneddies)] for _ in range(5)])




if __name__ == '__main__':
    main()