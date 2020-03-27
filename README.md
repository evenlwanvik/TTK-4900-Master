# TTK-4900-Master
TTK4900 - Engineering Cybernetics, Master's Thesis

## CNN 

### Predictions

![cnn_predicitons](/images/predicted_grid.png)
Format: ![](url)

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
  
## MATLAB GUI

One of the main concerns with the simple interface used for choosing training samples in python was that the algorithm only spits out the areas that has a very high likelihood of containing divergent flow. The Okuobu Weiss and R2 confidence level (how much the variation of dependent variables are explained by the independent variable of certain characteristics of the possible eddies compared to an ideal Gaussian eddy) spits out the same large eddies when found for grids seperated by only a few days. Although I had high hopes for the semi-automatic method, it does not provide a diverse enough pool of samples.

The GUI is a stand-alone application in MATLAB with variables and objects that are linked to the gui window. By one can use buttons to maneuver between datasets, i.e. the full grid, smaller windows (subgrids) of the full grid, and draw rectangles that holds the frame of a potential training data sample. There are four tabs showing different representations of the subgrid, one with both sea surface level (ssl) and ocean current velocity vectors, one that shows the phase angle of the ocean current, one for only the ocean current velocity, and the last one shows the ssl and vectors for the full grid. 

Whenever you have chosen a rectangle, it is displayed in a smaller figure in the upper right corner, which means you can analyze a smaller representation of the plot. Another thing that happens is that a popup window is shown, where you can decide wether the chosen frame has the label cyclone, anti-cyclone or nothing, or you can simply delete it. The rectangles deemed to be a training sample will stick around on the plot, with the color indicating its label; red for cyclone (high pressure area), blue for anti-cyclone (low pressure area), and simply black for the sampels labeled as "nothing".

The netcdf files has to be kept in "TBA". Before the application is run for the first time, the "set_config" has to be run to initiate application's internal counters for netcdf file (dataset) and what window you're on, such that you can simply exit the application, and continue where you dropped of the last time.
