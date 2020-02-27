addpath('C:\Master\TTK-4900-Master\Matlab')
% Function for custom popup window for choosing label
import popup.*

% Root figure for "app"
f = figure;
% Initialize figure (app) data
setappdata(f, 'rectangles', []); % Rectangle corner coordinates
setappdata(f, 'sampleID', 0); % Id of current sample
setappdata(f, 'rectPlotObj', []);  % The plotted rectangles 

% All files
dirPath = 'C:/Master/data/cmems_data/global_10km/2016/'; %gets directory
%myFiles = dir(fullfile(dirPath,'*.nc')); %gets all wav files in struct 

dirPath = 'C:/Master/data/cmems_data/global_10km/2016/'; %gets directory
myFiles = dir(fullfile(dirPath,'*.nc')); %gets all wav files in struct 

% Delete h5 file if already exists
h5Path = 'C:/Master/TTK-4900-Master/data/training_data/2016/h5/ssl.h5'
if exist(h5Path, 'file')==2
  delete(h5Path);
end

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
[~, ch] = contourf(axPrimary,lon,lat,z',60); 

%plotNewRect(~,~,ch1)
nextBtn = uicontrol(f,'callback', @(src,eventdata)plotNextDataset(f), 'Position',[20 5 60 20], 'string', 'Next');
createRectBtn = uicontrol(f,'callback',@(src,eventdata)createRect(f, ch, axPrimary, axSecondary),'Position',[100 5 80 20], 'string', 'CreateRect');
%saveRectBtn = uicontrol(f,'callback',@(src,eventdata)saveRect(f, ch),'Position',[200 5 70 20], 'string', 'SaveRect');
%deleteLatestRectBtn = uicontrol(f,'callback',@(src,eventdata)deleteLatestRect(f),'Position',[290 5 110 20], 'string', 'DeleteLatestRect');

function createRect(f, ch1, axPrimary, axSecondary)
    % Increase ID by one
    id = getappdata(f, 'sampleID');
    setappdata(f, 'sampleID', id + 1);

    % f is the figure (app) and ch1 is the handle for primary axis
    rect = getrect(axPrimary);
    rect(3) = rect(1) + rect(3);
    rect(4) = rect(2) + rect(4);
    fprintf('Creating rectangle: %.2f %.2f %.2f %.2f\n', rect)
    
    % Set the window
    xIdx = find(ch1.XData >= rect(1) & ch1.XData <= rect(3)); 
    yIdx = find(ch1.YData >= rect(2) & ch1.YData <= rect(4));
    
    % Append latest rectangle corner coordinates
    appendLatest(f, 'rectangles', rect)
    
    ssl = ch1.ZData(yIdx,xIdx); % (row,col)
    
    % Plot section in secondary axis
    [~, ch2] = contourf(axSecondary, ch1.XData(xIdx), ch1.YData(yIdx), ssl, 60);
    ch2.LevelList = ch1.LevelList; 

    % Show the section in the main axis, if you want to.
    latestRect = rectangle(axPrimary,'Position',[rect(1),rect(2),range([rect(1),rect(3)]),range([rect(2),rect(4)])]);
    
    % Spawn popup upon creation of rectangle
    switch popup()
        case 'No'
        case 'cyclone'
            % Append latest plotted rectangle object
            appendLatest(f, 'rectPlotObj', latestRect, ssl') % Tranpose for (lon,lat)
            saveSample(f, 1) % 1 is label for cyclone
        case 'antiCyclone'
            % Append latest plotted rectangle object
            appendLatest(f, 'rectPlotObj', latestRect, ssl')
            saveSample(f, -1)
        case 'nothing'
            % Append latest plotted rectangle object
            appendLatest(f, 'rectPlotObj', latestRect, ssl')
            saveSample(f, 0)
        case 'delete'
            deleteLatestRect(f)  
    end
end

function deleteLatestRect(f)
    % Delete latest created rectangle in main plot
    rectPlotObj = getappdata(f, 'rectPlotObj');
    delete(rectPlotObj(end));
    deleteLatest(f, 'rectPlotObj'); 
    % Delete indeces stored (latest row)
    deleteLatest(f, 'rectangles');
    % Reduce sample ID by one
    id = getappdata(f, 'sampleID');
    setappdata(f, 'sampleID', id - 1);
end

function saveSample(f, label, data)
    id = getappdata(f, 'sampleID');
    dataName = '/sample_data_' + string(id);
    labelName = '/sample_data_' + string(id);
    % Create h5 file for saving eddy data
    fprintf('Saving sample nr %i with label = %i with dimension: (%i, %i)\n', id, label, size(data)');
    h5create(h5Path, dataName, [length(xIdx), length(yIdx)])
    h5write(h5Path, dataName, ssl');
    h5create(h5Path, labelName, 1)
    h5write(h5Path, labelName, label');
end

function plotNextDataset(f)
    sample_name = '/sample_' + string(getappdata(f, 'sampleID'));
    %data = readtable('C:/Master/TTK-4900-Master/data/training_data/2016/csv/test.csv');
    data = h5read(h5Path, sample_name);
    fprintf('Dimension of loaded rectangle: (%i, %i)\n', size(data));
end

function appendLatest(f, name, latest)
    a = getappdata(f, name);
    a = cat(1, a, latest);
    setappdata(f, name, a); 
end

function deleteLatest(f, name)
    a = getappdata(f, name);
    a(end,:) = [];
    setappdata(f, name, a);
end











function saveRectbleh(f, ch1)
    r = getappdata(f, 'rectangles');
    [nrows, ncols] = size(r);
    % Iterate over rows
    
    %ssl = {};
    for ii = 1:nrows
        rect = r(ii,:);
        fprintf('Saving rectangle: %.2f %.2f %.2f %.2f\n', rect);
        
        % Set the window
        xIdx = ch1.XData >= rect(1) & ch1.XData <= rect(3); 
        yIdx = ch1.YData >= rect(2) & ch1.YData <= rect(4);
        
       
        
        ssl = ch1.ZData(yIdx,xIdx); % (row,col)
        
        setName = '/sample_' + string(ii);
        % Create h5 file for saving eddy data
        fprintf('Dimension of rectangle: (%i, %i)\n', size(ssl));
        h5create('C:/Master/TTK-4900-Master/data/training_data/2016/h5/ssl.h5', setName', [size(xIdx), size(yIdx)])
        h5write('C:/Master/TTK-4900-Master/data/training_data/2016/h5/ssl.h5', setName', ssl);
        
    end
    
    
    % 'WriteMode', append is possible
    %h5write('C:/Master/TTK-4900-Master/data/training_data/2016/h5/ssl.h5', '/myTestSet', ssl);
    %save('ssl')
    %csvwrite('C:/Master/TTK-4900-Master/data/training_data/2016/csv/test.csv', ssl)
    %save( 'C:/Master/TTK-4900-Master/data/training_data/2016/h5/h5test.mat', 'ssl', '-v7.3' )
    
end