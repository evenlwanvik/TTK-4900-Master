f = figure;

% All files
dirPath = 'C:/Master/data/cmems_data/global_10km/2016/'; %gets directory
%myFiles = dir(fullfile(dirPath,'*.nc')); %gets all wav files in struct 

dirPath = 'C:/Master/data/cmems_data/global_10km/2016/'; %gets directory
myFiles = dir(fullfile(dirPath,'*.nc')); %gets all wav files in struct 

clf()

axPrimary = axes('Units','Normalize','Box','on'); 
axSecondary = axes('Units','Normalize','Box','on'); 
scale = 0.3;  % percentage of original size
axSecondary.Position(3:4) = scale * axSecondary.Position(3:4); 
axSecondary.Position(1:2) = axPrimary.Position(1:2) + axPrimary.Position(3:4)*(1-scale); 

% Open netcdf variables
fName = myFiles(1).name;
fpath = fullfile(dirPath, fName);
fprintf(1, 'Now reading %s\n', fpath);
lon = ncread(fpath,'longitude') ; nx = length(lon) ; 
lat = ncread(fpath,'latitude') ; ny = length(lat) ; 
time = ncread(fpath,'time') ;

z = ncread(fpath,'zos',[1 1 1],[nx ny 1]);
[~, ch] = contourf(axPrimary,lon,lat,z',30); 

%plotNewRect(~,~,ch1)
nextBtn = uicontrol(f,'callback', @(src,eventdata)plotNextDataset(ch), 'Position',[20 5 60 20], 'string', 'Next');
rectBtn = uicontrol(f,'callback',@(src,eventdata)createRect(f, ch, axPrimary, axSecondary),'Position',[100 5 80 20], 'string', 'CreateRect');
rectBtn = uicontrol(f,'callback',@(src,eventdata)saveRect(f),'Position',[200 5 70 20], 'string', 'SaveRect');

setappdata(f, 'rectangles', [[]]);

function createRect(f, ch1, axPrimary, axSecondary)
    % f is the figure (app) and ch1 is the handle for primary axis
    rect = getrect;
    rect(3) = rect(1) + rect(3);
    rect(4) = rect(2) + rect(4);
    r = getappdata(f, 'rectangles');
    r = cat(1, r, rect);
    disp(r)
    setappdata(f, 'rectangles', r);
    
    % Set the window
    xIdx = ch1.XData >= r(1) & ch1.XData <= r(3); 
    yIdx = ch1.YData >= r(2) & ch1.YData <= r(4);
    
    % Plot section in secondary axis
    [~, ch2] = contourf(axSecondary, ch1.XData(xIdx), ch1.YData(yIdx), ch1.ZData(yIdx,xIdx), 30);
    ch2.LevelList = ch1.LevelList; 
    caxis(axSecondary, caxis(axPrimary));
    axis(axPrimary);
    axis(axSecondary);
    % Show the section in the main axis, if you want to.
    rectangle(axPrimary,'Position',[rect(1),rect(2),range([rect(1),rect(3)]),range([rect(2),rect(4)])]);
end

function saveRect(f)

end

function plotNextDataset(~,~)
    
end