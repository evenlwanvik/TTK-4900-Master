
fName = 'D:/Master/TTK-4900-Master/data/training_data/2016/h5/rcnn/ds1_sample_15.h5'
%h5disp(fName, '/box_idxs')

data = h5read(fName, '/data');
%H5F.close(h5f);

ssl = data(:,:,1);
uvel = data(:,:,2);
vvel = data(:,:,3);

figure()
contourf(ssl',50); 
hold('on')
vectorPlot = quiver(uvel', vvel','color',[0 0 0]); 
hold('off')

box_idxs = hdf5read(fName, '/box_idxs')
labels = hdf5read(fName, '/labels');

for i=1:length(box_idxs)
    x = box_idxs(1,:,i);
    y = box_idxs(2,:,i);
    bounds = [x(1),y(1),range([x(1),x(2)]),range([y(1),y(2)])];
    if labels(i)==1
        rectangle('Position',bounds,'lineWidth', 3,'EdgeColor',[0.8500 0.3250 0.0980])
    else
        rectangle('Position',bounds,'lineWidth', 3,'EdgeColor',[0 0.3 0.8510])
    end
end