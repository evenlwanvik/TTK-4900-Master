function [outArr] = reshapeForSliding(nLonFull, nLatFull, nLonWindow, nLatWindow)
%Returns a reshaped sliding window of the input array
%   The input array (grid) is reshaped with respect to the defined number of
%   longitudinal and latitudinal cells of the desired window/grid. Since
%   the full input grid is not that important for this application, the
%   last cells 

if nargin<3
	nLonWindow = 200;
elseif nargin<4
    nLatWindow = 120;
end

% Create a 2d grid of indeces
idxs = zeros(nLonFull, nLatFull, 2);


% Reshape idx array to 3d of desired windows
nWindowLon = floor(nLonFull/nLonWindow); % Numer of windows in lon dimension
lonMax = nWindowLon*nLonWindow; % The last possible lon index for full grid
nWindowLat = floor(nLatFull/nLatWindow); % Numer of windows in lon dimension
latMax = nWindowLat*nLatWindow; % The last possible lat index for full grid
idxs = idxs(1:lonMax, 1:latMax, :); % Just include the compatible indexes of full grid
nWindows = lonMax*latMax/(nLonWindow*nLatWindow); % Number of windows we then have

for ii=1:nLonFull
    for jj=1:nLatFull
        idxs(ii,jj,1) = ii; 
        idxs(ii,jj,2) = jj; 
    end
end
outArr = reshape(idxs, nWindows, nLonWindow, nLatWindow, 2); % Reshape into windows

disp(size(outArr))

end
%%
function skrrrrrt
winCtr = 1; % Window counter
for ii=1:nWindowLon
    for jj=1:nWindowLat
        for x=1:nLonWindow
            for y=1:nLatWindow
                
                outArr(x,y,1,winCtr) = (ii-1)*nLonWindow + x;
                outArr(x,y,2,winCtr) = (jj-1)*nLatWindow + y;
            end
        end
    end
    winCtr = winCtr + 1; 
end

%dims = arr.shape

counter = 1;
for x=1:nLonWindow:nLonFull
    for y=1:nLatWindow:nLatFull
        outArr(:,y,1,counter) = transpose(x : (x+nLonWindow-1));
        outArr(x,:,2,counter) = y : (y+nLonWindow-1);
    end
    counter = counter + 1;
end

end

