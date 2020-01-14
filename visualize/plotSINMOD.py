
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# single file
dataDIR = 'C:\Master\data\sCoast_surface_data/SINMOD_samp.nc'
ds = xr.open_dataset(dataDIR)

sst = ds.get(['temperature'])
lon = ds.get(['longitude'])
lat = ds.get(['latitude'])

test = sst.isel(xc=0, yc=0)
#test = sst.isel(time=0)
#print(test)
#test.plot()
#test.plot.pcolormesh()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9,3))
ds.xc.plot(ax=ax1)
ds.yc.plot(ax=ax2)