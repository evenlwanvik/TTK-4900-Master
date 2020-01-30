import xarray as xr
from math import radians, cos, sin, asin, sqrt, pi
import numpy as np
import time

# Eddy detection algorithm
def detect_eddies(lon,lat,uvel,vvel): 

    ######################################################################
    #  Prepare variables
    ######################################################################
    
    print("--- Calculating Okubo-Weiss ---")
    start_time = time.time()

    # Transpose from u(lat,lon) to u(lon,lat)
    uvel = uvel.transpose()
    vvel = vvel.transpose()
    # Fill the masked values with 0.0 for landmass
    uvel.set_fill_value(0.0)
    vvel.set_fill_value(0.0) 
    
    # Ocean mask with "True" for ocean cells
    ocean_mask = ~uvel.mask 
    n_ocean_cells = uvel.count()
    nx,ny = uvel.shape

    # Compute cartesian distances for derivatives, in m
    R = 6378e3

    x = np.zeros((nx,ny))
    y = np.zeros((nx,ny))

    for i in range(0,nx):
        for j in range(0,ny):
            x[i,j] = 2.*pi*R*cos(lat[j]*pi/180.)*lon[i]/360.
            y[i,j] = 2.*pi*R*lat[j]/360.

    dx,dy,grid_area = grid_cell_area(x,y)

    ######################################################################
    #  Compute Okubo-Weiss
    ######################################################################

    # Fill in mask with 0.0 to enable further calculations
    uvel = uvel.filled(0.0)


    # velocity derivatives
    dudx, dudy = central_diff_2d(uvel,x,y)
    dvdx, dvdy = central_diff_2d(vvel,x,y)

    # strain and vorticity
    normal_strain = dudx - dvdy
    shear_strain = dudy + dvdx
    vorticity = dvdx - dudy

    # compute OW
    OW_raw = normal_strain**2 + shear_strain**2 - vorticity**2
    OW_mean = OW_raw.sum() / n_ocean_cells
    OW_std = np.sqrt( np.sum( np.multiply(ocean_mask,(OW_raw-OW_mean))**2 ) / n_ocean_cells ) # standard deviation
    OW = OW_raw / OW_std

    print("--- Finished after %f seconds ---" % (time.time() - start_time))
    return OW, ocean_mask

def central_diff_2d(a,x,y):
# Take the first derivative of a with respect to x and y using
# centered central differences. 
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

    return dadx, dady

## Creates grid #####################################################
def grid_cell_area(x,y):
# Given 2D arrays x and y with grid cell locations, compute the
# area of each cell.
    
    nx,ny = x.shape
    dx = np.zeros((nx,ny))
    dy = np.zeros((nx,ny))
    
    for j in range(0,ny):
        dx[0,j] = x[1,j] - x[0,j]
        for i in range(1,nx-1):
            dx[i,j] = (x[i+1,j] - x[i-1,j]) / 2.0
        dx[nx-1,j] = x[nx-1,j] - x[nx-2,j]
    
    for i in range(0,nx):
        dy[i,0] = y[i,1] - y[i,0]
        for j in range(1,ny - 1):
            dy[i,j] = (y[i,j+1] - y[i,j-1]) / 2.0
        dy[i,ny-1] = y[i,ny-1] - y[i,ny-2]
    
    A = np.multiply(dx,dy)
    return (dx,dy,A)

def cartesian_distance(lon1, lat1, lon2, lat2):
    # Haversine formula to find distance between two points on a sphere
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # haversine formula 
    dlon = dlat = 80000 # meter
    # Radius of earth in kilometers is 6371
    dist = 6371 * c
    return dist




