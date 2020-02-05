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
import os
import io
import sys



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

    # Create eddy images for each day in datase
    for day, time in enumerate(t):
    #for day in range(1):

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
                    

            dirPath = 'C:/Master/TTK-4900-Master/images/'+dateStr+'/'
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

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

            guiEvent, guiValues = show_figure(fig)
            if guiEvent=='Yes':
                savedImgCounter = savedImgCounter + 1 
                imPath = dirPath + title + f"_{savedImgCounter}.png"   
                plt.savefig(imPath, bbox_inches='tight')

                sst_train.append([sst_eddy, cyclone]) # [data, label]
                ssl_train.append([ssl_eddy, cyclone]) 
                uvel_train.append([uvel_eddy, cyclone]) 
                vvel_train.append([vvel_eddy, cyclone]) 
                phase_train.append([phase_eddy, cyclone]) 
                

                print(f"---++++++ Saving image {eddyId}")   
            elif guiEvent=='No': 
                print(f"---++++++ Discarding image {eddyId}")

    print(f"--- Compressing training data")

    sst_train = np.array(sst_train)
    ssl_train = np.array(ssl_train)
    uvel_train = np.array(uvel_train)
    vvel_train = np.array(vvel_train)
    phase_train = np.array(phase_train)

    print(sst_train.shape)
    # Save data as compressed numpy array
    savez_compressed(f'{args.savedir}/sst_train.npz', sst_train)
    savez_compressed(f'{args.savedir}/ssl_train.npz', ssl_train)
    savez_compressed(f'{args.savedir}/uvel_train.npz', uvel_train)
    savez_compressed(f'{args.savedir}/vvel_train.npz', vvel_train)
    savez_compressed(f'{args.savedir}/phase_train.npz', phase_train)

    print(f"--- Training data complete")


if __name__ == '__main__':
    main()
