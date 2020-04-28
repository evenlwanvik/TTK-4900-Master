import numpy as np

def deriv1_central_diff_2D(a,x,y):
    # Take the first derivative of a with respect to x and y using
    # centered central differences. The variable a is a 3D field.   
    nx,ny = a.shape
    dadx = np.zeros((nx,ny))                            
    dady = np.zeros((nx,ny))
    
    for j in range(0,ny):
        dadx[0,j] = (a[1,j] - a[0,j]) / (x[1,j] - x[0,j])
        for i in range(1,nx-1):
            dadx[i,j] = (a[i+1,j] - a[i-1,j]) / (x[i+1,j] - x[i-1,j])
        dadx[nx-1,j] = (a[nx-1,j] - a[nx-2,j]) / (x[nx-1,j] - x[nx-2,j])
    
    for i in range(0,nx):
        dady[i,0]=(a[i,1] - a[i,0]) / (y[i,1] - y[i,0])
        for j in range(1,ny-1):
            dady[i,j]=(a[i,j+1] - a[i,j-1]) / (y[i,j+1] - y[i,j-1])
        dady[i,ny-1]=(a[i,ny-1] - a[i,ny-2]) / (y[i,ny-1] - y[i,ny-2])
    return dadx,dady 

def calc_OW(lon,lat,uvel,vvel,OW_start):  

    ########################################################################
    # Initialize variables
    ########################################################################
    
    # Since they are masked arrays (in the mask, True = NaN value), we can fill the masked values with 0.0 to describe land
    uvel.set_fill_value(0.0)
    vvel.set_fill_value(0.0)

    # Create an ocean mask which has value True at ocean cells.
    ocean_mask = ~uvel.mask
    n_ocean_cells = uvel.count()
    
    nx,ny= uvel.shape
    
    # Compute cartesian distances for derivatives, in m
    R = 6378e3
    
    x = np.zeros((nx,ny))
    y = np.zeros((nx,ny))
                 
    print("here")

    for i in range(0,nx):
        for j in range(0,ny):
            if lon.ndim > 1:
                x[i,j] = 2.*np.pi*R*np.cos(lat[i,j]*np.pi/180.)*lon[i,j]/360.
                y[i,j] = 2.*np.pi*R*lat[i,j]/360.
            else:
                x[i,j] = 2.*np.pi*R*np.cos(lat[j]*np.pi/180.)*lon[i]/360.
                y[i,j] = 2.*np.pi*R*lat[j]/360.

    # Gridcell area
    #dx,dy,grid_area = grid_cell_area(x,y)
    
    ########################################################################
    #  Compute Okubo-Weiss
    ########################################################################
    
    uvel = uvel.filled(0.0)
    vvel = vvel.filled(0.0)
    
    # velocity derivatives
    du_dx,du_dy = deriv1_central_diff_2D(uvel,x,y)
    dv_dx,dv_dy = deriv1_central_diff_2D(vvel,x,y)
    # strain and vorticity
    normal_strain = du_dx - dv_dy
    shear_strain = du_dy + dv_dx
    vorticity = dv_dx - du_dy
    
    # Compute OW, straight and then normalized with its standart deviation
    OW_raw = normal_strain ** 2 + shear_strain ** 2 - vorticity ** 2
    OW_mean = OW_raw.sum() / n_ocean_cells
    OW_std = np.sqrt(np.sum((np.multiply(ocean_mask,(OW_raw - OW_mean)) ** 2)) / n_ocean_cells)
    OW = OW_raw / OW_std
    
    # We create a mask with the possible location of eddies, meaning OW<-0.2
    OW_mask = np.zeros(OW.shape,dtype=int)
    OW_mask[np.where(OW < OW_start)] = 1

    # Seperate masks for cyclone and anti-cyclone depending on the vorticity polarity and magnitude     
    cyc_mask = np.where(vorticity < -1e-12, OW_mask, 0)
    acyc_mask = np.where(vorticity > 1e-12, OW_mask, 0)
    

    return OW, vorticity, OW_mask, cyc_mask, acyc_mask
