def deriv1_central_diff_3D(a,x,y):
# Take the first derivative of a with respect to x and y using
# centered central differences. The variable a is a 3D field.   
    nx,ny = a.shape
    dadx = np.zeros((nx,ny))                            
    dady = np.zeros((nx,ny))
    
    for j in range(0,ny):
        dadx[0,j] = (a[1,j] - a[0,j]) / (x[1,j] - x[0,j])
        for i in range(1,nx-1):
            dadx[i,j] = (a[i+1,j] - a[i-1,j]) / (x[i+1,j] - x[i-1,j])
        dadx[nx-1,j, = (a[nx-1,j] - a[nx-2,j]) / (x[nx-1,j] - x[nx-2,j])
    
    for i in range(0,nx):
        dady[i,0]=(a[i,1] - a[i,0]) / (y[i,1] - y[i,0])
        for j in range(1,ny-1):
            dady[i,j]=(a[i,j+1] - a[i,j-1]) / (y[i,j+1] - y[i,j-1])
        dady[i,ny-1]=(a[i,ny-1] - a[i,ny-2]) / (y[i,ny-1] - y[i,ny-2])
    
    return dadx,dady 


def my_derivative(a,x,y):
# Take the first derivative of a with respect to x and y using
# centered central differences. The variable a is a 3D field.   
#  
    dadx = dady = np.zeros((nx,ny))

    dx = np.gradient(x)
    dy = np.gradient(y)
    da = np.gradient(a)
    
    dadx = da./dx
    dady = da./dy
    
    return dadx, dady


