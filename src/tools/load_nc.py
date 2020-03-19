# Load netCDF4 data

import netCDF4 as nc4 

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
    return (ds,t,lon,lat,depth,u,v,sst,ssl)
    ds.close() 

    '''
    import netCDF4 as nc4 
    fp = "D:/Master/data/cmems_data/global_10km/2016/full/phys_noland_2016_001.nc"
    ds = nc4.Dataset(fp, 'r', format='NETCDF4') 
    test = ds.variables['zos'][:]
    '''