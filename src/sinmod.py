import os
os.chdir('c:')
import iris
#os.chdir('D:\master\TTK-4900-Master\src')
import matplotlib.pyplot as plt
from tools.load_nc import load_nc_cmems, load_nc_sinmod
from tools.grid_conversion import polar_lonlat_to_xy, polar_xy_to_lonlat
import numpy as np
from cmems_download import download
#from mpl_toolkits.basemap import Basemap

from sklearn.externals import joblib # To save scaler
from keras.models import load_model
import cv2



#import cartopy.crs as ccrs
#import cartopy

sinmod_fpath = 'D:/Master/data/cmems_data/sinmod/samples_2017.04.27_nonoverlap.nc'
cmems_fpath = 'D:/Master/data/cmems_data/global_10km/2016/noland/phys_noland_2016_001.nc'
cmems_full_fpath = 'D:/Master/data/cmems_data/global_10km/2016/full/phys_noland_2016_001.nc'

def closest_coordinate(lon, lat, lon_target, lat_target):
    '''Find abs distance of all points in grid and extract the closest, 
    if grid is very large, one should use a 2d search tree nearest neighbour'''
    a = abs(lat-lat_target) + abs(lon-lon_target)
    i,j = np.unravel_index(a.argmin(), a.shape)
    return i,j

# Test cv2 image and sliding window movement on smaller grid
def main():

    latitude = [50, 55]
    longitude = [-24, -12]

    lat_bnds = [68, 75]
    lon_bnds = [-4, 6]
    #download.download_nc(longitude, latitude)

    # cmems data
    lon_cmems,lat_cmems,sst_cmems,ssl_cmems,uvel_cmems,vvel_cmems =  load_nc_cmems(cmems_full_fpath)

    x_cmems, y_cmems = polar_lonlat_to_xy(lon_cmems, lat_cmems, 90, 6371, 0.0167, 1)
    x_cmems=x_cmems.values.T.dot(1e3); y_cmems=y_cmems.values.T.dot(1e3)

    xflat = x_cmems.flatten()
    yflat = y_cmems.flatten()
    avgXres = np.mean([ abs(xflat[i+1]-xflat[i]) for i in range(len(xflat)-1) ])
    avgYres = np.mean([ abs(yflat[i+1]-yflat[i]) for i in range(len(yflat)-1) ])
    print(avgXres)
    print(avgYres)

    lon_idxs = np.where((lon_cmems >= lon_bnds[0]) & (lon_cmems <= lon_bnds[1]))[0]
    lat_idxs = np.where((lat_cmems >= lat_bnds[0]) & (lat_cmems <= lat_bnds[1]))[0]

    # sinmod data
    xc_sinmod,yc_sinmod,lon_sinmod,lat_sinmod,sst_sinmod,ssl_sinmod,uvel_sinmod,vvel_sinmod =  load_nc_sinmod(sinmod_fpath)
    # Find the indexes of corners
    bottomleft = closest_coordinate(lon_sinmod, lat_sinmod, lon_bnds[0], lat_bnds[0])
    topleft = closest_coordinate(lon_sinmod, lat_sinmod, lon_bnds[0], lat_bnds[1])
    topright = closest_coordinate(lon_sinmod, lat_sinmod, lon_bnds[1], lat_bnds[1])
    bottomright = closest_coordinate(lon_sinmod, lat_sinmod, lon_bnds[1], lat_bnds[0])

    x_idxs = range(bottomleft[0],topright[0])
    y_idxs = range(bottomleft[1],topright[1])


    #lon, lat = polar_xy_to_lonlat(xc, yc, 90, 6371, 0.0167, 1)

    #x_bnds = [900000, 1600000]
    #y_bnds = [1500000, 2200000]
    # Find all longitudes that are within the latitude limits
    #x_idxs = np.where((xc_sinmod.values >= x_bnds[0]) & (xc_sinmod.values <= x_bnds[1]))[0]
    #y_idxs = np.where((yc_sinmod.values >= y_bnds[0]) & (yc_sinmod.values <= y_bnds[1]))[0]

    xc_sinmod = np.array(xc_sinmod[x_idxs])
    yc_sinmod = np.array(yc_sinmod[y_idxs])
    lon_sinmod = np.array([[lon_sinmod[i,j] for j in y_idxs] for i in x_idxs])
    lat_sinmod = np.array([[lat_sinmod[i,j] for j in y_idxs] for i in x_idxs])
    ssl_sinmod = np.array([[ssl_sinmod[i,j] for j in y_idxs] for i in x_idxs])
    uvel_sinmod = np.array([[uvel_sinmod[i,j] for j in y_idxs] for i in x_idxs])
    vvel_sinmod = np.array([[vvel_sinmod[i,j] for j in y_idxs] for i in x_idxs])

    lon_bnds = [lon_sinmod.min(), lon_sinmod.max()]
    lat_bnds = [lat_sinmod.min(), lat_sinmod.max()]
    lon_idxs = np.where((lon_cmems >= lon_bnds[0]) & (lon_cmems <= lon_bnds[1]))[0]
    lat_idxs = np.where((lat_cmems >= lat_bnds[0]) & (lat_cmems <= lat_bnds[1]))[0]

    x_cmems = np.array([[x_cmems[i,j] for j in lat_idxs] for i in lon_idxs])
    y_cmems = np.array([[y_cmems[i,j] for j in lat_idxs] for i in lon_idxs])
    lon_cmems = np.array(lon_cmems[lon_idxs])
    lat_cmems = np.array(lat_cmems[lat_idxs])
    ssl_cmems = np.array([[ssl_cmems[i,j] for j in lat_idxs] for i in lon_idxs])
    uvel_cmems = np.array([[uvel_cmems[i,j] for j in lat_idxs] for i in lon_idxs])
    vvel_cmems = np.array([[vvel_cmems[i,j] for j in lat_idxs] for i in lon_idxs])


    #fig, ax = plt.subplots(2,1,figsize=(12, 10), projection=ccrs.PlateCarree())
    #crs = ccrs.RotatedPole(pole_longitude=58, pole_latitude=90)
    crs = ccrs.NorthPolarStereo(central_longitude=58)
    ax1 = plt.subplot(2, 1, 1)
    ax1.contour(x_cmems, y_cmems, ssl_cmems,levels=80,cmap='rainbow',projection=crs)
    ax1.quiver(x_cmems, y_cmems, uvel_cmems, vvel_cmems)#, scale=5)

    ax2 = plt.subplot(2, 1, 2)
    ax2.contour(xc_sinmod, yc_sinmod, ssl_sinmod.T,levels=80,cmap='rainbow',projection=crs)
    ax2.quiver(xc_sinmod, yc_sinmod, uvel_sinmod.T, vvel_sinmod.T)#, scale=3)
    
    plt.show()
    exit()


def analyze_velocities(x,y,uvel,vvel):
    plt.figure()
    ax1 = plt.subplot(1, 3, 1)
    ax1.contour(x, y, uvel.T,levels=80,cmap='rainbow')
    ax2 = plt.subplot(1, 3, 2)
    ax2.contour(x, y, vvel.T,levels=80,cmap='rainbow')
    ax3 = plt.subplot(1, 3, 3)
    ax3.quiver(x, y, uvel.T, vvel.T, scale=3)

import cnn

def test_model():

    # cmems data
    lon_cmems,lat_cmems,sst_cmems,ssl_cmems,uvel_cmems,vvel_cmems =  load_nc_cmems(cmems_full_fpath)
    x_cmems, y_cmems = polar_lonlat_to_xy(lon_cmems, lat_cmems, 0, 6371, 0.0167, 1)
    x_cmems=x_cmems.values.T.dot(1e3); y_cmems=y_cmems.values.T.dot(1e3)
    xflat = x_cmems.flatten()
    yflat = y_cmems.flatten()
    avgXres = np.mean([ abs(xflat[i+1]-xflat[i]) for i in range(len(xflat)-1) ])
    avgYres = np.mean([ abs(yflat[i+1]-yflat[i]) for i in range(len(yflat)-1) ])

    # sinmod
    xc_sinmod,yc_sinmod,lon_sinmod,lat_sinmod,sst_sinmod,ssl_sinmod,uvel_sinmod,vvel_sinmod =  load_nc_sinmod(sinmod_fpath)

    #x_bnds = [754800, 1507600]
    #y_bnds = [994000, 1490000]
    x_bnds = [765000, 850000]
    y_bnds = [1390000, 1450000]
    #x_bnds = [865000, 950000]
    #y_bnds = [1490000, 1550000] 

    x_idxs = np.where((xc_sinmod.values >= x_bnds[0]) & (xc_sinmod.values <= x_bnds[1]))[0]
    y_idxs = np.where((yc_sinmod.values >= y_bnds[0]) & (yc_sinmod.values <= y_bnds[1]))[0]

    xc_sinmod = np.array(xc_sinmod[x_idxs])
    yc_sinmod = np.array(yc_sinmod[y_idxs])

    xflat = xc_sinmod.flatten()
    yflat = yc_sinmod.flatten()
    avgXres = np.mean([ abs(xflat[i+1]-xflat[i]) for i in range(len(xflat)-1) ])
    avgYres = np.mean([ abs(yflat[i+1]-yflat[i]) for i in range(len(yflat)-1) ])

    lon_sinmod = np.array([[lon_sinmod[i,j] for j in y_idxs] for i in x_idxs])
    lat_sinmod = np.array([[lat_sinmod[i,j] for j in y_idxs] for i in x_idxs])
    ssl_sinmod = np.array([[ssl_sinmod[i,j] for j in y_idxs] for i in x_idxs])
    uvel_sinmod = np.array([[uvel_sinmod[i,j] for j in y_idxs] for i in x_idxs])
    vvel_sinmod = np.array([[vvel_sinmod[i,j] for j in y_idxs] for i in x_idxs])


    # Create the coordinate system from grid_mapping data given in SINMOD dataset
    target_cs = iris.coord_systems.Stereographic(
        central_lat=90, 
        central_lon=58, 
        false_easting=3254000.0,
        false_northing=2560000.0,
        tue_scale_lat=60)

    uvel_new, vvel_new = iris.analysis.cartography.rotate_winds(uvel, vvel, target_cs)



    model_fpath='D:/master/models/2016/cnn_mult_velocities_9652.h5'
    scaler_fpath = "D:/master/models/2016/cnn_norm_scaler.pkl"
    scaler = joblib.load(scaler_fpath)
    clf = load_model(model_fpath)

    winW, winH = int(44), int(24)
    nlon, nlat = uvel_sinmod.shape
    data = [ssl_sinmod, uvel_sinmod, vvel_sinmod]

    print(lon_sinmod.shape)
    analyze_velocities(lon_sinmod, lat_sinmod, uvel_sinmod.T, vvel_sinmod.T)
    analyze_velocities(xc_sinmod,yc_sinmod,uvel_sinmod,vvel_sinmod)
    analyze_velocities(xc_sinmod,yc_sinmod,uvel_new,vvel_new)
    
    print(uvel_sinmod.shape)

    data_scaled = []
    for i in [1,2]:
        scaled = cv2.resize(data[i], dsize=(winH, winW), interpolation=cv2.INTER_CUBIC)
        data_scaled.append(scaled)
        scaled = scaled.flatten()
        scaled = scaler[i].transform([scaled])
        data_scaled.append(scaled.reshape((winW, winH)))
    
    X_cnn = np.zeros((1,winW,winH,2))
    for lo in range(winW): # Row
        for la in range(winH): # Column
            for c in range(2): # Channels
                X_cnn[0,lo,la,c] = data_scaled[c][lo,la]

    xc_sinmod = range(len(data_scaled[0]))
    yc_sinmod = range(len(data_scaled[0][0]))
    analyze_velocities(xc_sinmod,yc_sinmod,data_scaled[0],data_scaled[1])
    plt.show()
    exit()

    prob = clf.predict(X_cnn)
    print(prob)
    exit()

    #crs = ccrs.NorthPolarStereo()
    ax1 = plt.subplot(1, 1, 1)
    ax1.contour(ssl_sinmod.T,levels=80,cmap='rainbow')
    ax1.quiver(uvel_sinmod.T, vvel_sinmod.T, scale=5)
    plt.show()
    
    data = [xc_sinmod, yc_sinmod, ssl_sinmod, uvel_sinmod, vvel_sinmod]
    model_fpath='D:/master/models/2016/cnn_mult_velocities_9652.h5'
    cnn.test_model(custom_data=data, model_fpath=model_fpath)

if __name__ == '__main__':
    #main() 
    test_model()
