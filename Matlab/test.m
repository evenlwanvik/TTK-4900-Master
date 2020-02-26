dirPath = 'C:/Master/data/cmems_data/global_10km/2016/'; %gets directory
myFiles = dir(fullfile(dirPath,'*.nc')); %gets all wav files in struct
   
fig = uifigure('Name','My Figure');

figHandle = uifigure('Name','Measured Data', 'position', [200 100 1400 900]);
p = uipanel(f,'Position',[20 20 196 135]);
b = uibutton(p,'Position',[11 40 140 22],'Text','Send');



%%

dirPath = 'C:/Master/data/cmems_data/global_10km/2016/'; %gets directory
myFiles = dir(fullfile(dirPath,'*.nc')); %gets all wav files in struct 

fig = uifigure('Name','My Figure', 'position', [200 100 1400 900]);

p1 = uipanel('Parent',f,'BorderType','none'); 
p1.Title = {'CT to Blue Fluoro'; 'Bead Position Error vs Radial Bead Distance'}; 
p1.TitlePosition = 'centertop'; 
p1.FontSize = 11;
p1.FontWeight = 'bold';

global rectangles;
global rectId;
newRectButton(f);

for k = 1:2%length(myFiles) 
    
    % Create figure
    %figHandle = figure('Name','Measured Data', 'position', [200 100 1400 900]);
    clf()
   
    axPrimary = axes('Units','Normalize','Box','on'); 
    axSecondary = axes('Units','Normalize','Box','on'); 
    scale = 0.3;  % percentage of original size
    axSecondary.Position(3:4) = scale * axSecondary.Position(3:4); 
    axSecondary.Position(1:2) = axPrimary.Position(1:2) + axPrimary.Position(3:4)*(1-scale); 

    % Open netcdf variables
    fName = myFiles(k).name;
    fpath = fullfile(dirPath, fName);
    fprintf(1, 'Now reading %s\n', fpath);
    lon = ncread(fpath,'longitude') ; nx = length(lon) ; 
    lat = ncread(fpath,'latitude') ; ny = length(lat) ; 
    time = ncread(fpath,'time') ;

    z = ncread(fpath,'zos',[1 1 1],[nx ny 1]);
    [~, ch] = contourf(axPrimary,lon,lat,z',30); 
    
    % Choose section to isolate
    xSection = [-30, -25];  % x-bounds
    ySection = [50, 52];  % y-bounds
    
    % Get index values of section
    xIdx = ch.XData >= xSection(1) & ch.XData <= xSection(2); 
    yIdx = ch.YData >= ySection(1) & ch.YData <= ySection(2);
    
    % Plot section in secondary axis
    [~, ch2] = contourf(axSecondary, ch.XData(xIdx), ch.YData(yIdx), ch.ZData(yIdx,xIdx), 30);
    ch2.LevelList = ch.LevelList; 
    %caxis(axSecondary, caxis(axPrimary));
    axis(axPrimary);
    axis(axSecondary);
    % Show the section in the main axis, if you want to.
    rectangle(axPrimary,'Position',[xSection(1),ySection(1),range(xSection),range(ySection)]);
    
    % Just loop untill 
    %while ishandle(figHandle)
    %end
end

function newRectButton(fig)
    % Create a figure window
    %fig = uifigure('Name','Button Panel', 'Position',  [100, 100, 300, 200]);
    % Create a push button
    p2 = uipanel('Parent',fig,'BorderType','none'); 
    btn = uibutton(p2,'push',...
                   'Position',[115,200, 200, 222],...
                   'ButtonPushedFcn', @(btn,event) newRectButtonPushed());
end

function newRectButtonPushed()
    % Create a new rectangle and add to list
    global rectangles;
    global rectId;
    rect = getrect;
    rectangles = [rectangles, [rect]];
    disp(rectangles);
    rectId = rectId + 1;
end