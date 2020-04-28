% Current values in the file are stored as east-north components.

% We need to read these and rotate into model currents:

x_bnds = [765000, 850000];
y_bnds = [1390000, 1450000];
idx_start = [floor(x_bnds(1)/4000), floor(y_bnds(1)/4000)]
idx_len = [floor((x_bnds(2)-x_bnds(1))/4000), floor((y_bnds(2)-y_bnds(1))/4000)]

sinmod_fpath = 'D:/Master/data/cmems_data/sinmod/samples_2017.04.27_nonoverlap.nc';
ncdisp(sinmod_fpath)
u_east = double(ncread(sinmod_fpath,'u_east',[idx_start,1,1],[idx_len,1,1]));
v_north = double(ncread(sinmod_fpath,'v_north',[idx_start,1,1],[idx_len,1,1]));
ssl = double(ncread(sinmod_fpath,'elevation',[idx_start,1],[idx_len,1]));
xc = double(ncread(sinmod_fpath,'xc',idx_start(1),idx_len(1)));
yc = double(ncread(sinmod_fpath,'yc',idx_start(2),idx_len(2)));
longitude = double(ncread(sinmod_fpath,'gridLons',idx_start,idx_len));
latitude = double(ncread(sinmod_fpath,'gridLats',idx_start,idx_len));

phi = -(pi/180)*(58-longitude); % This is the angle by which we need to rotate currents to get the oriented with the model grid.

uField = zeros(idx_len);

vField = zeros(idx_len);

for i=1:numel(uField)
    [u_east(i); v_north(i)]
    rotMat = [cos(phi(i)) -sin(phi(i)); sin(phi(i)) cos(phi(i))]
    rotated = rotMat*[u_east(i); v_north(i)]
    break;
    

    uField(i) = rotated(1);

    vField(i) = rotated(2);

end

figure()
contour(xc,yc,ssl')
hold('on')
quiver(xc,yc,uField',vField')
hold('off')

