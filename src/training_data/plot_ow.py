from eddies import load_netcdf4,eddy_detection,dataframe_eddies,plot_eddies,julianh2gregorian
import os
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from math import cos, pi
import numpy as np
import xarray as xr
import datetime

#name of the netCDF data file
filepath = 'C:/Master/data/cmems_data/global_10km/phys_noland_001.nc'
filepath = 'D:/master/data/cmems_data/global_10km/2016/noland/realtime/phys_noland_2016_001.nc'
# load data
(ds,t,lon,lat,depth,uvel,vvel,sst,ssl) =  load_netcdf4(filepath)

# Confidence level, usually 90%
R2_criterion = 0.9

# OW value at which to begin the evaluation of R2, default was -1, want to use -8 to be absolutely sure
OW_start = -3.0

# Number of local minima to evaluate using R2 method.
# Set low (like 20) to see a few R2 eddies quickly.
# Set high (like 1e5) to find all eddies in domain.
max_evaluation_points = 2000 #set to 2000 to capture avery minima, there should be around 1800

# Minimum number of cells required to be identified as an eddie.
min_eddie_cells = 1 # set to 3 to be coherent with the use of the R2 method, 3 points seems like a reasonable minimun for a correlation 

# z-level to plot.  Usually set to 0 for the surface.
k_plot = 0

day = 0
lon,lat,u,v,vorticity,OW,OW_eddies,eddie_census,nEddies,circulation_mask = eddy_detection(
    lon,lat,depth,uvel,vvel,day,R2_criterion,OW_start,max_evaluation_points,min_eddie_cells)

myOW = OW[:,:,0] # just a quick fix to remove depth...
OW_noneddies = np.zeros(myOW.shape,dtype=int)
OW_noneddies[np.where(myOW > -0.7)] = 1 # non-eddy are as 1
OW_noneddies = np.ma.masked_equal(OW_noneddies, 0) # rest is masked

fig = plt.subplots(figsize=(10, 7))
#plt.contourf(lon, lat, OW.T[0], 15, cmap='bwr')
plt.contourf(lon, lat, OW_noneddies.T, 20)#, cmap='Greys')#, alpha=0.3)
plt.show()