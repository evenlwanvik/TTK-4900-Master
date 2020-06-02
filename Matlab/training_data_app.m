addpath('C:/Users/47415/Master/TTK-4900-Master/Matlab')
% Function for custom popup window for choosing label
import popup.*
load('C:/Users/47415/Master/TTK-4900-Master/Matlab/config.mat')

% Root figure for "app"
f = figure; clf()
hTabGroup = uitabgroup('Parent',f);
sslTab = uitab('Parent',hTabGroup, 'Title','ssl');
set(hTabGroup, 'SelectedTab',sslTab);
setappdata(f, 'sslTab', sslTab); 
phaseTab = uitab('Parent',hTabGroup, 'Title','phase'); setappdata(f, 'phaseTab', sslTab);
vectorTab = uitab('Parent',hTabGroup, 'Title','vector field'); setappdata(f, 'vectorTab', sslTab);
fullTab = uitab('Parent',hTabGroup, 'Title','Full grid (ssl)'); setappdata(f, 'fullTab', fullTab);

% Initialize figure (app) data
setappdata(f, 'rectangles', []); % Rectangle corner coordinates
setappdata(f, 'sampleID', config('sampleID')); % Id of current sample
setappdata(f, 'datasetID', config('datasetID')); % Id of current netcdf dataset, the config file registers where we left off
setappdata(f, 'windowID', config('windowID')); % Id of current window of the dataset
setappdata(f, 'rectPlotObj', []);  % The plotted rectangles 
setappdata(f, 'windowSize', [144,60]); % lon / lat size

% Directory to all files
dirPath = 'D:/Master/data/cmems_data/global_10km/2016/full/'; %gets directory
setappdata(f, 'dirPath', dirPath);
setappdata(f, 'ncfiles', dir(fullfile(dirPath,'*.nc'))); %gets all wav files in struct 
% I will be saving training samples as individual matlab cell arrays
setappdata(f, 'storePath', "C:/Users/47415/Master/TTK-4900-Master/data/training_data/2016/h5/")

% Set primary and secondary axes to plot on
axPrimary(1) = axes('Parent',sslTab,'Units','Normalize','Box','on');
axPrimary(2) = axes('Parent',phaseTab,'Units','Normalize','Box','on');
axPrimary(3) = axes('Parent',vectorTab,'Units','Normalize','Box','on');
axPrimary(4) = axes('Parent',fullTab,'Units','Normalize','Box','on');
%axPrimary = axes('Units','Normalize','Box','on'); 
setappdata(f, 'axPrimary', axPrimary);
axSecondary(1) = axes('Parent',sslTab, 'Units','Normalize','Box','on');
axSecondary(2) = axes('Parent',phaseTab, 'Units','Normalize','Box','on'); 
axSecondary(3) = axes('Parent',vectorTab, 'Units','Normalize','Box','on'); 
for ii=1:3
    scale = 0.3;  % percentage of original size
    axSecondary(ii).Position(3:4) = scale * axSecondary(ii).Position(3:4); 
    axSecondary(ii).Position(1:2) = axPrimary(ii).Position(1:2) + axPrimary(ii).Position(3:4)*(1-scale);
end
setappdata(f, 'axSecondary', axSecondary);
setappdata(f, 'currentTab', 1);
setappdata(f, 'tabName', ["ssl", "phase", "vector"]);

chooseDataset(f, 'Next')

%plotNewRect(~,~,ch1)
uicontrol(f,'callback', @(src,eventdata)chooseDataset(f, 'Prev'), 'Position',[20 5 60 20], 'string', 'PrevDS');
uicontrol(f,'callback', @(src,eventdata)chooseDataset(f, 'Next'), 'Position',[80 5 60 20], 'string', 'NextDS');

uicontrol(f,'callback', @(src,eventdata)plotDatasetWindow(f, 'Prev'), 'Position',[150 5 70 20], 'string', 'PrevWin');
uicontrol(f,'callback', @(src,eventdata)plotDatasetWindow(f, 'Next'), 'Position',[220 5 70 20], 'string', 'NextWin');

uicontrol(f,'callback',@(src,eventdata)createRect(f),'Position',[300 5 80 20], 'string', 'CreateRect');
uicontrol(f,'callback',@(src,eventdata)deleteLatestRect(f),'Position',[380 5 110 20], 'string', 'DeleteLatestRect');

% --- Plot either next or previous netcdf dataset
function chooseDataset(f, next_or_prev) % load
    ncFiles = getappdata(f, 'ncfiles'); % gets all wav files in struct
    if strcmp(next_or_prev, 'Prev')
        id = getappdata(f, 'datasetID') - 1; 
        if (id < 0); disp('Dataset ID < 0'); 
        else; setappdata(f, 'datasetID', id); end;
    else % if 'Next'
        id = getappdata(f, 'datasetID') + 1; setappdata(f, 'datasetID', id);
    end
    load('C:/Users/47415/Master/TTK-4900-Master/Matlab/config.mat'); config('datasetID') = id;
    save 'C:/Users/47415/Master/TTK-4900-Master/Matlab/config.mat' config;
    % Open netcdf variables
    fName = ncFiles(id).name;
    dPath = getappdata(f, 'dirPath');
    fPath = [dPath, fName];
    setappdata(f, 'fPath', fPath)
    fprintf(1, 'Now reading %s, datasetID %d\n', fPath, id);
    
    %lon/lat
    lon = ncread(fPath,'longitude') ; nx = length(lon) ; 
    setappdata(f, 'nx', nx)
    lat = ncread(fPath,'latitude') ; ny = length(lat) ; 
    setappdata(f, 'ny', ny)  
    
    % Reset the window ID before showing next dataset
    load('C:/Users/47415/Master/TTK-4900-Master/Matlab/config.mat'); config('windowID') = 0;
    save 'C:/Users/47415/Master/TTK-4900-Master/Matlab/config.mat' config;
    setappdata(f, 'windowID', 0)
    plotDatasetWindow(f, next_or_prev)

    ssl = ncread(fPath,'zos',[1,1,1],[nx,ny,1]);
    uvel = ncread(fPath,'uo',[1,1,1,1],[nx,ny,1,1]);
    vvel = ncread(fPath,'vo',[1,1,1,1],[nx,ny,1,1]);
    ax = getappdata(f, 'axPrimary');
    contourf(ax(4),lon,lat,ssl',100); 
    hold(ax(4),'on')
    x = quiver(ax(4),lon,lat,uvel',vvel','color',[0 0 0]); 
    set(x,'AutoScale','on', 'AutoScaleFactor', 2)
    hold(ax(4),'off')
    
    h5Path = getappdata(f, 'storePath'); zipPath = h5Path + "training_data.zip";
    % Check if zipped file containing training data exists
    if isfile(zipPath)
        % Extract content of file
    	unzip(zipPath, h5Path);
    end
    % If any h5 files exists, zip them
    if ~isempty(dir(h5Path + "*.h5"))
        zip(h5Path + "training_data.zip", "*.h5", h5Path)
        delete(h5Path + "*.h5")
    end
end

% --- Plot either next or previous window of full dataset
function plotDatasetWindow(f, next_or_prev) % load
    if strcmp(next_or_prev, 'Prev')
        id = getappdata(f, 'windowID') - 1; 
        if (id <= 0); disp('Window ID < 0'); 
        else; setappdata(f, 'windowID', id); end   
    else % if 'Next'
        id = getappdata(f, 'windowID') + 1; setappdata(f, 'windowID', id);
    end
    if (id <= 0); disp('Window ID <= 0'); return; end
    load('C:/Users/47415/Master/TTK-4900-Master/Matlab/config.mat'); config('windowID') = id;
    save 'C:/Users/47415/Master/TTK-4900-Master/Matlab/config.mat' config;
    
    windowSize = getappdata(f, 'windowSize');
    nx = getappdata(f, 'nx'); ny = getappdata(f, 'ny');
    nWindowLon = floor(nx/windowSize(1));
    nWindowLat = floor(ny/windowSize(2));

    if id > nWindowLon*nWindowLat
        fprintf(1, 'Max nr of windows are %d\n', nWindowLon*nWindowLat); return;   
    end
    
    % lonStart and latStart represents sliding window decided by the
    % window size and current index
    lonStart = floor(0.1^mod(id-1,nWindowLon)) + mod(id-1,nWindowLon) * windowSize(1);
    latStart = floor(0.1^mod(floor((id-1)/nWindowLat),nWindowLat)) + floor((id-1)/nWindowLat) * windowSize(2);
    
    fprintf(1, '\nWindow id %d - lonStart: %d with length %d\n', id, lonStart, windowSize(1));
    fprintf(1, 'Window id %d - latStart: %d with length %d \n\n', id, latStart, windowSize(2));

    fPath = getappdata(f, 'fPath');
    lon = ncread(fPath,'longitude', lonStart, windowSize(1));
    lat = ncread(fPath,'latitude', latStart, windowSize(2));
    ssl = ncread(fPath,'zos',[lonStart,latStart,1],[windowSize(1),windowSize(2),1]);
    sst = ncread(fPath,'thetao',[lonStart,latStart,1,1],[windowSize(1),windowSize(2),1,1]);
    uvel = ncread(fPath,'uo',[lonStart,latStart,1,1],[windowSize(1),windowSize(2),1,1]);
    vvel = ncread(fPath,'vo',[lonStart,latStart,1,1],[windowSize(1),windowSize(2),1,1]);
    phase = rad2deg(atan(vvel./uvel));
    setappdata(f, 'uvel', uvel) % need velocities for seondary axis
    setappdata(f, 'vvel', vvel)
    setappdata(f, 'phase', phase) % Since we won't be plotting sst, we'll need to save it for the save
    setappdata(f, 'sst', sst) % Since we won't be plotting sst, we'll need to save it for the save
    
    ax = getappdata(f, 'axPrimary');
    % Plot the ssl window  
    [~, ch(1)] = contourf(ax(1),lon,lat,ssl',150); 
    hold(ax(1),'on')
    x = quiver(ax(1),lon,lat,uvel',vvel','color',[0 0 0]); 
    set(x,'AutoScale','on', 'AutoScaleFactor', 2)
    hold(ax(1),'off')
    % Plot the phase window (second tab)
    [~, ch(2)] = contourf(ax(2),lon,lat,phase');
    % Plot the velocity vector window (last tab)
    ch(3) = quiver(ax(3),lon,lat,uvel',vvel','color',[0 0 0],'linewidth',5);  
    set(ch(3),'AutoScale','on', 'AutoScaleFactor', 2)
    
    % Set the channel for the primary axes
    setappdata(f, 'ch', ch); 
end

% --- Create an rectangle and plot in secondary axis
function createRect(f)
    axPrim = getappdata(f, 'axPrimary'); % Holds primary figures
    axSec = getappdata(f, 'axSecondary'); 
    ch = getappdata(f, 'ch'); % Channel holds axes of the primary plots
    
    % f is the figure (app) and ch1 is the handle for primary axis
    rect = getrect(axPrim(1));
    rect(3) = rect(1) + rect(3);
    rect(4) = rect(2) + rect(4);
    fprintf('Displaying rectangle: %.2f %.2f %.2f %.2f\n', rect)
    
    % Set the window
    xIdx = find(ch(1).XData >= rect(1) & ch(1).XData <= rect(3)); 
    yIdx = find(ch(1).YData >= rect(2) & ch(1).YData <= rect(4));
    lon = ch(1).XData(xIdx); lat = ch(1).YData(yIdx);
    
    ssl = ch(1).ZData(yIdx,xIdx);
    phase = ch(2).ZData(yIdx,xIdx);
        
    uvel = getappdata(f, 'uvel'); uvel = uvel(xIdx, yIdx);
    vvel = getappdata(f, 'vvel'); vvel = vvel(xIdx, yIdx);
    
    % Secondary axis ssl tab plot
    contourf(axSec(1), lon, lat, ssl); 
    hold(axSec(1),'on')
    vectorPlot = quiver(axSec(1), lon, lat, uvel', vvel','color',[0 0 0]); 
    set(vectorPlot,'AutoScale','on', 'AutoScaleFactor', 2)
    hold(axSec(1),'off') 

    % Secondary axis phase tab plot
    contourf(axSec(2), lon, lat,  phase);

    % Simply copy quiver axis to the vector tab as they are equal
    x = quiver(axSec(3), lon, lat, uvel', vvel','color',[0 0 0]);
    set(x,'AutoScale','on', 'AutoScaleFactor', 2)

    latestRect(1) = rectangle(axPrim(1),'Position',[rect(1),rect(2),range([rect(1),rect(3)]),range([rect(2),rect(4)])], 'lineWidth', 3);
    latestRect(2) = rectangle(axPrim(2),'Position',[rect(1),rect(2),range([rect(1),rect(3)]),range([rect(2),rect(4)])], 'lineWidth', 3);
    latestRect(3) = rectangle(axPrim(3),'Position',[rect(1),rect(2),range([rect(1),rect(3)]),range([rect(2),rect(4)])], 'lineWidth', 3);

    setappdata(f, 'lon_window', lon);
    setappdata(f, 'lat_window', lat);
    setappdata(f, 'ssl_window', ssl);
    sst = getappdata(f, 'sst');
    setappdata(f, 'sst_window', sst(xIdx, yIdx)'); % sst, uvel and vvel has not been transposed yet
    setappdata(f, 'uvel_window', uvel');
    setappdata(f, 'vvel_window', vvel');
    setappdata(f, 'phase_window', phase);
    
    % Spawn popup upon creation of rectangle
    switch popup()
        case 'No'
            disp('Discarding the last rectangle')
            delete(latestRect)
        case 'Cyclone'
            disp('Appending a sample of a cyclone')
            latestRect(1).EdgeColor = [0.8500 0.3250 0.0980]; % red/orange color for cyclone
            latestRect(2).EdgeColor = [0.8500 0.3250 0.0980]; % red/orange color for cyclone
            latestRect(3).EdgeColor = [0.8500 0.3250 0.0980]; % red/orange color for cyclone
            % Append latest plotted rectangle object
            appendLatest(f, 'rectPlotObj', latestRect) % Tranpose for (lon,lat)
            saveSample(f, 1) % 1 is label for cyclone
            appendLatest(f, 'rectangles', rect)
        case 'Anti-Cyclone'
            latestRect(1).EdgeColor = [0 0.3 0.8510]; % Blue color for cyclone
            latestRect(2).EdgeColor = [0 0.3 0.8510]; % Blue color for cyclone
            latestRect(3).EdgeColor = [0 0.3 0.8510]; % Blue color for cyclone
            disp('Appending a sample of an anti-cyclone')
            % Append latest plotted rectangle object
            appendLatest(f, 'rectPlotObj', latestRect)
            saveSample(f, -1)
            appendLatest(f, 'rectangles', rect)
        case 'Nothing'
            disp('Appending a sample of nothing')
            % Append latest plotted rectangle object
            appendLatest(f, 'rectPlotObj', latestRect)
            saveSample(f, 0)
            appendLatest(f, 'rectangles', rect)
        case 'Delete'
            disp('Discarding the last sample')
            delete(latestRect);  
    end
end

% --- Discard the latest rectangle drawn in figure
function deleteLatestRect(f)
    disp('Removing latest sample')
    % Delete latest created rectangle in main plot
    rectPlotObj = getappdata(f, 'rectPlotObj');
    delete(rectPlotObj(end,:));
    deleteLatest(f, 'rectPlotObj'); 
    % Delete indeces stored (latest row)
    deleteLatest(f, 'rectangles');
    % Reduce sample ID by one
    id = getappdata(f, 'sampleID') - 1;
    setappdata(f, 'sampleID', id);
    dsId = getappdata(f, 'datasetID');
    % Delete latest sample csv file
    fName = "/ds" + string(dsId) + "_" + "sample" + "_" + string(id) + ".h5";
    h5Path = getappdata(f, 'storePath') + fName;
    delete(h5Path);
    load('C:/Users/47415/Master/TTK-4900-Master/Matlab/config.mat'); config('sampleID') = id;
    save 'C:/Users/47415/Master/TTK-4900-Master/Matlab/config.mat' config;
end

% --- Save rectangle as hdf5 training sample
function saveSample(f, label)
    id = getappdata(f, 'sampleID');
    dsId = getappdata(f, 'datasetID');
    setappdata(f, 'sampleID', id + 1);
    fName = "/ds" + string(dsId) + "_" + "sample" + "_" + string(id) + ".h5";
    savePath = getappdata(f, 'storePath') + fName;
    fprintf('Saving sample nr %i with label = %i and dimension: (%i, %i)\n', id, label, size(getappdata(f, 'ssl'))');

    load('C:/Users/47415/Master/TTK-4900-Master/Matlab/config.mat'); config('sampleID') = id;
    save 'C:/Users/47415/Master/TTK-4900-Master/Matlab/config.mat' config;
    
    %ssl
    data = getappdata(f, 'ssl_window');
    h5create(savePath,'/data/ssl',size(data));
    h5write(savePath,'/data/ssl',data);
    %sst
    data = getappdata(f, 'sst_window');
    h5create(savePath,'/data/sst',size(data));
    h5write(savePath,'/data/sst',data);
    %phase
    data = getappdata(f, 'phase_window');
    h5create(savePath,'/data/phase',size(data));
    h5write(savePath,'/data/phase',data); 
    %uvel
    data = getappdata(f, 'uvel_window'); 
    h5create(savePath,'/data/uvel',size(data));
    h5write(savePath,'/data/uvel',data);
    %vvel
    data = getappdata(f, 'vvel_window');
    h5create(savePath,'/data/vvel',size(data));
    h5write(savePath,'/data/vvel',data);
    % lon/lab
    data = getappdata(f, 'lon_window');
    h5create(savePath,'/coordinates/lon',size(data));
    h5write(savePath,'/coordinates/lon',data);
    data = getappdata(f, 'lat_window');
    h5create(savePath,'/coordinates/lat',size(data));
    h5write(savePath,'/coordinates/lat',data);
    %label
    h5create(savePath,'/label',1);
    h5write(savePath,'/label',label);  

end

% --- Generic function for appending latest addition to some variable bound to figure
function appendLatest(f, name, latest)
	a = getappdata(f, name);
    a = cat(1, a, latest);
    setappdata(f, name, a); 
end

% --- Generic function for deleting the latest appended variable bound to figure
function deleteLatest(f, name)
    a = getappdata(f, name);
    a(end,:) = [];
    setappdata(f, name, a);
end