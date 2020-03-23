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
    

def load_nc_data(fpath):
    ''' Simpler method '''
    ds = xr.open_dataset(fpath)

    lon = ds.longitude
    lat = ds.latitude
    # Mask NaN - indicating land
    sst = np.ma.masked_invalid(ds.thetao[0,0].T)
    ssl = np.ma.masked_invalid(ds.zos[0].T)
    uvel = np.ma.masked_invalid(ds.uo[0,0].T)
    vvel = np.ma.masked_invalid(ds.vo[0,0].T)
    ds.close() 
    return lon,lat,sst,ssl,uvel,vvel