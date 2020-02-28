A = rand(4340,2850);

nLon = 200; nLat = 120;
lonMax = floor(size(A,1)/nLon)*nLon; % The last possible lon index for full grid
latMax = floor(size(A,2)/nLat)*nLat; % The last possible lat index for full grid
A = A(1:lonMax, 1:latMax); % Just include the compatible indexes of full grid
nWindows = size(A,1)*size(A,2)/(nLon*nLat) % Number of windows we then have

x = reshape(A, nWindows, nLon, nLat); % Reshape into windows

%%

w = 200;
h = 100;
for ii=1:w:size(A,1) % Lon
    for jj=1:h:size(A,2) % Lat
        x = ii*jj : (w*h);
        reshape(x, w, h);
        fprintf('final lon = %d and lat = %d\n',x(w), x(h));
    end
  end