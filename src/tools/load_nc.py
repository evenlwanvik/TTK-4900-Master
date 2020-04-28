# Load netCDF4 data

import netCDF4 as nc4 
import numpy as np
import xarray as xr

def load_netcdf4(filename):
    ''' We omit the depth variable! '''
    ds = nc4.Dataset(filename, 'r', format='NETCDF4') 
    lon = ds.variables['longitude'][:]
    lat = ds.variables['latitude'][:]
    depth = ds.variables['depth'][:]
    # Load zonal and meridional velocity, in m/s
    u = ds.variables['uo'][:]
    v = ds.variables['vo'][:]
    # Load sea surface temperature
    sst = ds.variables['thetao'][:].squeeze(axis=1)
    # Load sea surface level
    ssl = ds.variables['zos'][:]
    # Load time in hours from 1950-01-01?
    t = ds.variables['time'][:]
    ds.close() 
    return (ds,t,lon,lat,depth,u,v,sst,ssl)
    

def load_nc_cmems(fpath):
    ''' Simpler method '''
    ds = xr.open_dataset(fpath)

    lon = np.array(ds.longitude)
    lat = np.array(ds.latitude)
    # Mask NaN - indicating land
    sst = np.ma.masked_invalid(ds.thetao[0,0].T)
    ssl = np.ma.masked_invalid(ds.zos[0].T)
    uvel = np.ma.masked_invalid(ds.uo[0,0].T)
    vvel = np.ma.masked_invalid(ds.vo[0,0].T)
    ds.close() 
    return lon,lat,sst,ssl,uvel,vvel


def load_nc_sinmod(fpath):
    ''' Simpler method '''
    ds = xr.open_dataset(fpath)
    #print(ds)
    xc = ds.xc
    yc = ds.yc
    lon = ds.gridLons.values.T
    lat = ds.gridLats.values.T
    # Mask NaN - indicating land
    sst = np.ma.mean(np.ma.masked_invalid(ds.temperature[:,0]), axis=0).T
    ssl = np.ma.mean(np.ma.masked_outside(ds.elevation, -4, 4), axis=0).T
    uvel = np.ma.mean(np.ma.masked_invalid(ds.u_east[:,0]), axis=0).T
    vvel = np.ma.mean(np.ma.masked_invalid(ds.v_north[:,0]), axis=0).T

    ds.close() 
    return xc, yc, lon,lat,sst,ssl,uvel,vvel