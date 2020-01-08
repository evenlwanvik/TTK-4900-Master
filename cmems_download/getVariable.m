function field = getVariable(ncid, name, start, dims, sample)

varid = netcdf.inqVarID(ncid, name);
try
    scale = netcdf.getAtt(ncid, varid, 'scale_factor');
catch
    scale = 1;
end
try
    offset = netcdf.getAtt(ncid, varid, 'add_offset');
catch 
    offset = 0;
end
fillVal = netcdf.getAtt(ncid, varid, '_FillValue');

field = netcdf.getVar(ncid, varid, [start(1) start(2) 0 sample], [dims(1) dims(2) 1 1],'double');
fillInds = field==fillVal;

field = offset + scale*double(field);
field(fillInds) = NaN;