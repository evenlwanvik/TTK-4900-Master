# TTK-4900-Master
TTK4900 - Engineering Cybernetics, Master's Thesis


## Semi-automated training data generator

I've created a semi-automated way of extracting annotated data from CMEMS global satellite data. 

## What does the algorithm do?
**TMDI;dr:** Uses the Okubo-Weiss parameter to determine what is eddies or not. A GUI then showcases the eddies to an who validates the results. The eddies are resized to fit a standard frame size and stored as a compressed numpy array.


1. **For each day:** 
    1. Runs the OW-R2 algorithm that outputs the eddy census, i.e. how many eddies, center of eddie, diameter, etc,. The OW-R2 algorithm will be explained in a seperate section TBA. 
    1. It actually runs the algorithm twice, but with different boundaries on the OW parameter; such that it returns the areas absent eddies for images classified as non-eddy.
    1. **For each eddy found:**
        1. Use the centre of the eddy and its width (longitude) and height (latitude) to find a rectangular grid encompassing the eddy.
        1. Use a simple GUI to showcase the eddies for an expert, who can either tell the program that this is an eddy or not.
        1. Append another axis with the given classifcation.
        1. If it is an eddy, append it to the training data, if not discard it entirelly.
    1. After n amount of valid eddies are found, find n non-eddy features.
        1. It uses the average frame size of eddies (or the largest frame found for this day) to divide the whole grid of satellite data in to subgrids with this frame size.
        1. For each subgrid (an abundant amount of them) it discards the ones which has a cell with a cell masked as an eddy.
        1. From the remaining subgrids, it extracts n random subgrids to be used as non-eddies in the training set.
2. Resize (cv2.resize(interpolate)) the image to a standard fram size (or the largest/average frame found for all days).
3. Save dataset as a compressed numpy array.
  

 
