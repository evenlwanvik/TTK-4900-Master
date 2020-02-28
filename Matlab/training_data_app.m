addpath('C:\Master\TTK-4900-Master\Matlab')
% Function for custom popup window for choosing label
import popup.*
load('C:\Master\TTK-4900-Master\Matlab\config.mat')

% Root figure for "app"
f = figure; clf()
hTabGroup = uitabgroup('Parent',f);
sslTab = uitab('Parent',hTabGroup, 'Title','ssl');
set(hTabGroup, 'SelectedTab',sslTab);
setappdata(f, 'sslTab', sslTab); 
phaseTab = uitab('Parent',hTabGroup, 'Title','phase'); setappdata(f, 'phaseTab', sslTab);
vectorTab = uitab('Parent',hTabGroup, 'Title','vector field'); setappdata(f, 'vectorTab', sslTab);

% Initialize figure (app) data
setappdata(f, 'rectangles', []); % Rectangle corner coordinates
setappdata(f, 'sampleID', 1); % Id of current sample
setappdata(f, 'datasetID', config('datasetID')); % Id of current netcdf dataset, the config file registers where we left off
setappdata(f, 'rectPlotObj', []);  % The plotted rectangles 

% Directory to all files
dirPath = 'C:/Master/data/cmems_data/global_10km/2016/'; %gets directory
setappdata(f, 'dirPath', dirPath);
setappdata(f, 'ncfiles', dir(fullfile(dirPath,'*.nc'))); %gets all wav files in struct 
% I will be saving training samples as individual matlab cell arrays
setappdata(f, 'storePath', 'C:/Master/TTK-4900-Master/data/training_data/2016/mlab/')

% Set primary and secondary axes to plot on
axPrimary(1) = axes('Parent',sslTab,'Units','Normalize','Box','on');
axPrimary(2) = axes('Parent',phaseTab,'Units','Normalize','Box','on');
axPrimary(3) = axes('Parent',vectorTab,'Units','Normalize','Box','on');
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

plotDataset(f, 'Next')

%plotNewRect(~,~,ch1)
nextBtn = uicontrol(f,'callback', @(src,eventdata)plotDataset(f, 'Next'), 'Position',[20 5 60 20], 'string', 'Next');
prevBtn = uicontrol(f,'callback', @(src,eventdata)plotDataset(f, 'Prev'), 'Position',[100 5 60 20], 'string', 'Prev');
createRectBtn = uicontrol(f,'callback',@(src,eventdata)createRect(f),'Position',[200 5 80 20], 'string', 'CreateRect');
%saveRectBtn = uicontrol(f,'callback',@(src,eventdata)saveRect(f, ch),'Position',[200 5 70 20], 'string', 'SaveRect');
deleteLatestRectBtn = uicontrol(f,'callback',@(src,eventdata)deleteLatestRect(f),'Position',[300 5 110 20], 'string', 'DeleteLatestRect');

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
    vectorPlot = quiver(axSec(1), lon, lat, uvel', vvel'); 
    %set(vectorPlot,'AutoScale','on', 'AutoScaleFactor', 2)
    hold(axSec(1),'off') 

    % Secondary axis phase tab plot
    contourf(axSec(2), lon, lat,  phase);

    % Simply copy quiver axis to the vector tab as they are equal
    x = copyobj(vectorPlot, axSec(3));
    set(x,'AutoScale','on', 'AutoScaleFactor', 2)

    latestRect(1) = rectangle(axPrim(1),'Position',[rect(1),rect(2),range([rect(1),rect(3)]),range([rect(2),rect(4)])], 'lineWidth', 3);
    latestRect(2) = rectangle(axPrim(2),'Position',[rect(1),rect(2),range([rect(1),rect(3)]),range([rect(2),rect(4)])], 'lineWidth', 3);
    latestRect(3) = rectangle(axPrim(3),'Position',[rect(1),rect(2),range([rect(1),rect(3)]),range([rect(2),rect(4)])], 'lineWidth', 3);
    
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
    delete(rectPlotObj(end));
    deleteLatest(f, 'rectPlotObj'); 
    % Delete indeces stored (latest row)
    deleteLatest(f, 'rectangles');
    % Reduce sample ID by one
    id = getappdata(f, 'sampleID') - 1;
    setappdata(f, 'sampleID', id);
    % Delete latest sample csv file
    dataName = '/sample_' + string(id);
    csvPath = getappdata(f, 'storePath') + dataName + '.csv';
    delete(csvPath);
end

% --- Save rectangle as csv training sample
function saveSample(f, label)
    id = getappdata(f, 'sampleID');
    setappdata(f, 'sampleID', id + 1);
    dataName = '/sample_' + string(id);
    savePath = getappdata(f, 'storePath') + dataName + '.csv';
    fprintf('Saving sample nr %i with label = %i and dimension: (%i, %i)\n', id, label, size(getappdata(f, 'ssl'))');
    cellArr = {'ssl', 'phase', 'vvel', 'uvel', 'y'; getappdata(f, 'ssl'), getappdata(f, 'phase'), getappdata(f, 'uvel'), getappdata(f, 'vvel'), label};
    cell2csv(savePath, cellArr);
end

% --- Plot either next or previous netcdf dataset
function plotDataset(f, next_or_prev)
    ncFiles = getappdata(f, 'ncfiles'); % gets all wav files in struct
    if strcmp(next_or_prev, 'Prev')
        id = getappdata(f, 'datasetID') - 1; setappdata(f, 'datasetID', id);
    else % if 'Next'
        id = getappdata(f, 'datasetID') + 1; setappdata(f, 'datasetID', id);
    end
    disp(id)
    load('C:\Master\TTK-4900-Master\Matlab\config.mat');
    config('datasetID') = id;
    save 'C:\Master\TTK-4900-Master\Matlab\config.mat' config;
    id = getappdata(f, 'datasetID'); setappdata(f, 'datasetID', id+1);
    % Open netcdf variables
    fName = ncFiles(id).name;
    dPath = getappdata(f, 'dirPath');
    fPath = [dPath, fName];
    fprintf(1, 'Now reading %s\n', fPath);
    lon = ncread(fPath,'longitude') ; nx = length(lon) ; 
    lat = ncread(fPath,'latitude') ; ny = length(lat) ; 
    %time = ncread(fPath,'time') ;
    tabNames = getappdata(f, 'tabName');
    
    % I don't know how to update the quiver plots of secondary axis
    % properly, so for now I have to set velocity variables in the app
    uvel = ncread(fPath,'uo',[1 1 1 1],[nx ny 1 1]);
    setappdata(f, 'uvel', uvel)
    vvel = ncread(fPath,'vo',[1 1 1 1],[nx ny 1 1]);
    setappdata(f, 'vvel', vvel)
    
    for ii=1:3
        ax = getappdata(f, 'axPrimary');
        if strcmp(tabNames(ii), 'ssl')
            ssl = ncread(fPath,'zos',[1 1 1],[nx ny 1]);
            [~, ch(ii)] = contourf(ax(ii),lon,lat,ssl',100); 
            hold(ax(ii),'on')
            x = quiver(ax(ii),lon,lat,uvel',vvel'); 
            set(x,'AutoScale','on', 'AutoScaleFactor', 2)
            hold(ax(ii),'off')
        elseif strcmp(tabNames(ii), 'phase')
            phase = rad2deg(atan(vvel./uvel));
            [~, ch(ii)] = contourf(ax(ii),lon,lat,phase');
            %contourcmap('jet',10,'colorbar','on','location','horizontal')
        elseif strcmp(tabNames(ii), 'vector')
            ch(ii) = quiver(ax(ii),lon,lat,uvel',vvel');  
            set(ch(ii),'AutoScale','on', 'AutoScaleFactor', 2)
        else 
            disp("No tabs chosen")
        end
        % Set the channel for the primary axes
        setappdata(f, 'ch', ch);
    end
end

% --- Generic function for appending latest addition to some variable bound to figure
function appendLatest(f, name, latest)
	a = getappdata(f, name);
    disp(length(latest))
    if (length(a) > 1)
        for ii=1:length(latest)
            a(ii) = cat(1, a(ii), latest(ii));
        end
    else
        a = cat(1, a, latest);
    end
    setappdata(f, name, a); 
end

% --- Generic function for deleting the latest appended variable bound to figure
function deleteLatest(f, name)
    a = getappdata(f, name);
    a(end,:) = [];
    setappdata(f, name, a);
end
