def create_subarray(lon,lat,uvel,vvel,OW_lim,nCells,nEddies):
    """ 
    Finds metrics such as index centroid and diameter of eddy 

    Parameters:
    ----------
    OW_lim : float
        OW value at which to create a binary mask of the map -> most likely neglecting the eddies 
    n_cells : tuple
        number of cells in the non-eddy grid 
    nEddies : int
        Number of eddies selected for training
        
    returns:
    ----------
    masked array of 

    """
