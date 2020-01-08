
ncid = netcdf.open('C:/Users/evenwa/Workspaces/Master/cmems_data/multiobs_001.nc');
varids = netcdf.inqVarIDs(ncid);
for i=1:length(varids)
   name = netcdf.inqVar(ncid, varids(i))
end
%%

depth = netcdf.getVar(ncid, netcdf.inqVarID(ncid, 'depth'));
latitude = netcdf.getVar(ncid, netcdf.inqVarID(ncid, 'latitude'));
longitude = netcdf.getVar(ncid, netcdf.inqVarID(ncid, 'longitude'));
sample = 1;

start = [0, 0];%[230, 65];
dims = [length(longitude), length(latitude)]; %[90, 45];

%sea_surface_temperature = getVariable(ncid, 'thetao', dims, 0);
%sea_surface_height = getVariable(ncid, 'zos', dims, 0);

sea_surface_temperature = getVariable(ncid, 'to', start, dims, 0);
sea_surface_height = getVariable(ncid, 'zo', start, dims, 0);
ugo = getVariable(ncid, 'ugo', start, dims, 0);
vgo = getVariable(ncid, 'vgo', start, dims, 0);
speed = sqrt(ugo.^2 + vgo.^2);

figure, 
subplot(2,3,1), pcolor(sea_surface_temperature'), shading flat, colorbar, title('Sea surface temperature');
subplot(2,3,2), pcolor(sea_surface_height'), shading flat, colorbar, title('Sea surface height');
subplot(2,3,4), pcolor(ugo'), shading flat, colorbar, title('U');
subplot(2,3,5), pcolor(vgo'), shading flat, colorbar, title('V');
subplot(2,3,6), pcolor(speed'), shading flat, colorbar, title('speed');