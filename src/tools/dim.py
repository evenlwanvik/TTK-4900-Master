import numpy as np

def dim(a):
    """" Recursively add length of subdimensions to list """
    if not isinstance(a,(list,np.ndarray)):
        return []
    return [len(a)] + dim(a[0])

def shape(a):
    """ Get the shape of list """
    return (np.array(a).shape)

def find_max_dim(a):
    """" Find the largest dimension of list or array """
    # Obs! This used to be dim(i[0]) from training_data for the train and label axis!
    return max( [dim(i) for i in a]) 


def find_avg_dim(a):
    """" Find the average frame size for both col and row, [0] since we want the training data, and not the label """
    # Obs! This used to be dim(i[0]) from training_data for the train and label axis!
    x = np.array([dim(i) for i in a])
    return x.mean(axis=0, dtype=int)