fName = 'D:/Master/data/cmems_data/global_10km/2016/full/phys_noland_2016_006.nc';

ncdisp(fName,'time')

dt = datetime(1950,1,1)+ hours(585804)

% last window labeled for normal CNN dataset 5:
% Window id 5 - lonStart: 576 with length 144
%Window id 5 - latStart: 1 with length 60 