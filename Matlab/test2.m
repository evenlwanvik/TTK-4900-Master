% Root figure for "app"
f = figure;
% The rectangles appdata will hold the coordinates of the frame
setappdata(f, 'rectangles', [[]]);

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
createRectBtn = uicontrol(f,'callback',@(src,eventdata)createRect(f, ch, axPrimary, axSecondary),'Position',[100 5 80 20], 'string', 'CreateRect');
saveRectBtn = uicontrol(f,'callback',@(src,eventdata)saveRect(f, ch),'Position',[200 5 70 20], 'string', 'SaveRect');
nextBtn = uicontrol(f,'callback', @(src,eventdata)plotNextDataset(ch), 'Position',[20 5 60 20], 'string', 'Next');

function createRect(f, ch1, axPrimary, axSecondary)
    % f is the figure (app) and ch1 is the handle for primary axis
    rect = getrect(axPrimary);
    rect(3) = rect(1) + rect(3);
    rect(4) = rect(2) + rect(4);
    fprintf('Creating rectangle: %.2f %.2f %.2f %.2f\n', rect)
    
    % Set the window
    xIdx = ch1.XData >= rect(1) & ch1.XData <= rect(3); 
    yIdx = ch1.YData >= rect(2) & ch1.YData <= rect(4);

    r = getappdata(f, 'rectangles');
    r = cat(1, r, rect);
    setappdata(f, 'rectangles', r);    
    
    % Plot section in secondary axis
    [~, ch2] = contourf(axSecondary, ch1.XData(xIdx), ch1.YData(yIdx), ch1.ZData(yIdx,xIdx), 30);
    ch2.LevelList = ch1.LevelList; 

    % Show the section in the main axis, if you want to.
    rectangle(axPrimary,'Position',[rect(1),rect(2),range([rect(1),rect(3)]),range([rect(2),rect(4)])]);
end

function saveRect(f, ch1)
    r = getappdata(f, 'rectangles');
    [nrows, ncols] = size(r);
    % Iterate over rows
    
    ssl = {};
    for ii = 1:nrows
        rect = r(ii,:);
        fprintf('Saving rectangle: %.2f %.2f %.2f %.2f\n', rect);
        
        % Set the window
        xIdx = ch1.XData >= rect(1) & ch1.XData <= rect(3); 
        yIdx = ch1.YData >= rect(2) & ch1.YData <= rect(4); 
        
        ssl = cat(3, ssl, ch1.ZData(yIdx,xIdx));
        
        setName = '/eddy_' + string(i)
        % Create h5 file for saving eddy data
        %h5create('C:/Master/TTK-4900-Master/data/training_data/2016/h5/ssl.h5', setName, 1)
        
    end
    
    
    % 'WriteMode', append is possible
    %h5write('C:/Master/TTK-4900-Master/data/training_data/2016/h5/ssl.h5', '/myTestSet', ssl);
    %save('ssl')
    %csvwrite('C:/Master/TTK-4900-Master/data/training_data/2016/csv/test.csv', ssl)
    %save( 'C:/Master/TTK-4900-Master/data/training_data/2016/h5/h5test.mat', 'ssl', '-v7.3' )
    
end

function plotNextDataset(ch)
    data = readtable('C:/Master/TTK-4900-Master/data/training_data/2016/csv/test.csv');
    %data = h5read('C:/Master/TTK-4900-Master/data/training_data/2016/h5/ssl.h5', '/myTestSet')
    disp(data)
end