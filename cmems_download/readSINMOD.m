ncid = netcdf.open('C:\Master\sCoast_surface_data/SINMOD_samp.nc');
varids = netcdf.inqVarIDs(ncid);
for i=1:length(varids)
   name = netcdf.inqVar(ncid, varids(i))
end



%%

depth = netcdf.getVar(ncid, netcdf.inqVarID(ncid, 'depth'));
latitude = netcdf.getVar(ncid, netcdf.inqVarID(ncid, 'gridLats'));
longitude = netcdf.getVar(ncid, netcdf.inqVarID(ncid, 'gridLons'));
sample = 1;

start = [0, 0];%[230, 65];
dims = [length(longitude), length(latitude)]; %[90, 45];

sea_surface_temperature = getVariable(ncid, 'temperature', start, dims, 0);
%sea_surface_temperature = netcdf.getVar(ncid, netcdf.inqVarID(ncid, 'temperature'));
%%
sea_surface_height = getVariable(ncid, 'depth', start, dims, 0);
%sea_surface_height = netcdf.getVar(ncid, netcdf.inqVarID(ncid, 'depth'));
%%
%ugo = getVariable(ncid, 'u-velocity', start, dims, 0);
u = netcdf.getVar(ncid, netcdf.inqVarID(ncid, 'u-velocity'));
%vgo = getVariable(ncid, 'v-velocity', start, dims, 0);
v = netcdf.getVar(ncid, netcdf.inqVarID(ncid, 'v-velocity'));
speed = sqrt(u.^2 + v.^2);

figure, 
subplot(2,3,1), pcolor(sea_surface_temperature(1)'), shading flat, colorbar, title('Sea surface temperature');
subplot(2,3,2), pcolor(sea_surface_height(1)'), shading flat, colorbar, title('Sea surface height');
subplot(2,3,4), pcolor(u(1)'), shading flat, colorbar, title('U');
subplot(2,3,5), pcolor(v(1)'), shading flat, colorbar, title('V');
subplot(2,3,6), pcolor(speed(1)'), shading flat, colorbar, title('speed');