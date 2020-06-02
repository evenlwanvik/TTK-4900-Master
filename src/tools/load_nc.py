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
    

def load_nc_sat(fpath):
    ''' Load CMEMS satellite measurements data '''
    ds = xr.open_dataset(fpath)

    lon = np.array(ds.longitude)
    lat = np.array(ds.latitude)
    # Mask NaN - indicating land
    sst = np.ma.masked_invalid(ds.thetao[0,0].T)
    ssl = np.ma.masked_invalid(ds.zos[0].T)
    sal = np.ma.masked_invalid(ds.so[0,0].T)
    uvel = np.ma.masked_invalid(ds.uo[0,0].T)
    vvel = np.ma.masked_invalid(ds.vo[0,0].T)
    ds.close() 
    return lon,lat,sst,ssl,sal,uvel,vvel


def load_nc_sinmod(fpath):
    ''' Simpler method '''
    ds = xr.open_dataset(fpath)
    xc = ds.xc
    yc = ds.yc
    depth = ds.depth
    lon = ds.gridLons.values.T
    lat = ds.gridLats.values.T
    # Mask NaN - indicating land
    sst = np.ma.mean(np.ma.masked_invalid(ds.temperature[:,0]), axis=0).T
    ssl = np.ma.mean(np.ma.masked_outside(ds.elevation, -4, 4), axis=0).T
    sal = np.ma.mean(np.ma.masked_invalid(ds.salinity[:,0]), axis=0).T
    uvel = np.ma.mean(np.ma.masked_invalid(ds.u_east[:,0]), axis=0).T
    vvel = np.ma.mean(np.ma.masked_invalid(ds.v_north[:,0]), axis=0).T

    ds.close() 
    return xc, yc, depth, lon,lat,sst,ssl,sal,uvel,vvel



def load_nc_insitu(fpath):
    ''' Load CMEMS satellite measurements data '''
    ds = xr.open_dataset(fpath)

    lon = np.array(ds.longitude)
    lat = np.array(ds.latitude)
    # Mask NaN - indicating land
    sst = np.ma.masked_invalid(ds.to[0,0].T)
    ssl = np.ma.masked_invalid(ds.zo[0,0].T)
    sal = np.ma.masked_invalid(ds.so[0,0].T)
    uvel = np.ma.masked_invalid(ds.ugo[0,0].T)
    vvel = np.ma.masked_invalid(ds.vgo[0,0].T) 
    ds.close() 
    return lon,lat,sst,ssl,sal,uvel,vvel, ds.time

#https://resources.marine.copernicus.eu/?option=com_csw&task=results?option=com_csw&view=details&product_id=MULTIOBS_GLO_PHY_NRT_015_001