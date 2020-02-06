import xarray as xr
from math import radians, cos, sin, asin, sqrt, pi
import numpy as np
import itertools
import time

# Eddy detection algorithm
def get_noneddies(t,lon,lat,uvel,vvel,sst,ssl,phase,OW_lim,nCells,nEddies,eddy_shape): 
    """ 
    Finds metrics such as index centroid and diameter of eddy 

    Parameters:
    ----------
    OW_lim : float
        OW value at which to create a binary mask of the map -> most likely neglecting the eddies 
    n_cells : tuple
        number of cells in the non-eddy grid 
    nEddies : int
        Number of eddies selected for training
    shape : tuple
        Array shape which will be used (lon,lat)

    returns:
    ----------
    masked array of 

    """

    ######################################################################
    #  Prepare variables
    ######################################################################
    
    print("--- Calculating Okubo-Weiss ---")
    start_time = time.time()

    # We transpose the data to fit with the algorithm provided, the correct order is uvel(lon,lat,depth) while the original from the netCDF is uvel(time,lat,lon,depth)
    uvel = uvel[day,:,:,:].transpose(2,1,0)
    vvel = vvel[day,:,:,:].transpose(2,1,0)
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

    for i,j in itertools.product(range(nx), range(ny)):
        x[i,j] = 2.*pi*R*cos(lat[j]*pi/180.)*lon[i]/360.
        y[i,j] = 2.*pi*R*lat[j]/360.

    dx,dy,grid_area = grid_cell_area(x,y)

    ######################################################################
    #  Compute Okubo-Weiss
    ######################################################################

    # Fill in mask with 0.0 to enable further calculations
    uvel = uvel.filled(0.0)
    vvel = vvel.filled(0.0)


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

    # We create a mask with the possible location of non-eddies, meaning OW>0, i.e. the opposite of when identifying an eddy.
    OW_noneddies = np.zeros(OW.shape,dtype=int)
    OW_noneddies[np.where(OW > OW_lim)] = 1

    print("--- Finished after %f seconds ---" % (time.time() - start_time))

    # Pick 50 random non-eddy images around the map

    ######################################################################
    #  Return the measurement data for grids that only contains either vorticity- or strain-dominated region.
    ######################################################################

    # Number of grids to create per day
    nGrids = int(nEddies/len(day))

    # Create standard size of grid for which to find random cell
    w = len(lon[0])
    h = len(lat[0])
    gridArea = w*h/nGrids # Area of each grid box
    gridSide = sqrt(gridArea) # Side length of each grid box
    n_lon = floor(w/boxSide)  # Number of boxes that fit along width
    n_lat = floor(h/boxSide)

    datasets = [sst, ssl, uvel, vvel, phase]
    for ds in datasets:
        for day, data in enumerate(ds):
            subgrids = create_subgrids(data, n_lon, n_lat)
            for grid in subgrids:
                rand_lon = np.random.randint(0,n_lon)
                rand_lat = np.random.randint(0,n_lat)
                rand_image = grid[rand_lon:eddy_shape[0]][rand_lat:eddy_shape[1]]
                print(rand_image.count)

    return 0#OW_noneddies, ocean_mask


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

def noneddy_training_data():
    """ Return the measurement data for grids that only contains either vorticity- or strain-dominated region. """



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




