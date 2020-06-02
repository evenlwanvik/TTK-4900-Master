from tools.compare_tools import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
from cnn import cnn_predict_grid
from skimage import measure
from tools.load_nc import load_nc_sat, load_nc_sinmod, load_nc_insitu
from matplotlib.patches import Rectangle
import pandas as pd
from scipy.interpolate import griddata, interp2d
from scipy.signal import resample
from tools import grid_conversion
from scipy.spatial import cKDTree
import os
import datetime as dt

def load_grid(x_bnds, y_bnds,
            sinmod_fpath='D:/master/data/compare/samples_2017.04.27_nonoverlap.nc', 
            sat_fpath='D:/master/data/compare/phys_2016_001.nc',
            model_fpath = 'D:/Master/models/2016/cnn_mult_velocities_9758_5_epoker.h5'):

    print(f"++ Loading grid within x: {x_bnds} and y {y_bnds}")

    # Load satellite netcdf data
    lons_sat,lats_sat,sst_sat,ssl_sat,sal_sat,uvel_sat,vvel_sat =  load_nc_sat(sat_fpath)
    x_sinmod,y_sinmod,_,lons_sinmod,lats_sinmod,sst_sinmod,ssl_sinmod,sal_sinmod,uvel_sinmod,vvel_sinmod =  load_nc_sinmod(sinmod_fpath)
    # Get SINMOD data

    # Get a polar steregraphic projection with cartesian irregular curvilinear grid for satellite data 
    x_sat_2d, y_sat_2d = grid_conversion.bl2xy(*np.meshgrid(lons_sat, lats_sat), FE=3254000, FN=2560000, xy_res=1, slon=58, SP=60)

    # Interpolate data and the curvilienar grid axes such that it has a higher resolution than SINMOD
    sst_sat = interp2d_masked(sst_sat, lons_sat, lats_sat, 4, 'linear')
    ssl_sat = interp2d_masked(ssl_sat, lons_sat, lats_sat, 4, 'linear')
    sal_sat = interp2d_masked(sal_sat, lons_sat, lats_sat, 4, 'linear')
    uvel_sat = interp2d_masked(uvel_sat, lons_sat, lats_sat, 4, 'linear')
    vvel_sat = interp2d_masked(vvel_sat, lons_sat, lats_sat, 4, 'linear')
    x_sat = interp2d_masked(x_sat_2d.T, lons_sat, lats_sat, 4, 'linear')
    y_sat = interp2d_masked(y_sat_2d.T, lons_sat, lats_sat, 4, 'linear')

    x_idxs = np.where((x_sinmod >= x_bnds[0]) & (x_sinmod <= x_bnds[1]))[0]
    y_idxs = np.where((y_sinmod >= y_bnds[0]) & (y_sinmod <= y_bnds[1]))[0]
    x_sinmod = np.ma.array(x_sinmod[x_idxs])
    y_sinmod = np.ma.array(y_sinmod[y_idxs])
    lons_sinmod = np.ma.array(lons_sinmod[x_idxs[0]:x_idxs[-1]+1, y_idxs[0]:y_idxs[-1]+1])
    lats_sinmod = np.ma.array(lats_sinmod[x_idxs[0]:x_idxs[-1]+1, y_idxs[0]:y_idxs[-1]+1])
    sst_sinmod = np.ma.array(sst_sinmod[x_idxs[0]:x_idxs[-1]+1, y_idxs[0]:y_idxs[-1]+1])
    ssl_sinmod = np.ma.array(ssl_sinmod[x_idxs[0]:x_idxs[-1]+1, y_idxs[0]:y_idxs[-1]+1])
    sal_sinmod = np.ma.array(sal_sinmod[x_idxs[0]:x_idxs[-1]+1, y_idxs[0]:y_idxs[-1]+1])
    uvel_sinmod = np.ma.array(uvel_sinmod[x_idxs[0]:x_idxs[-1]+1, y_idxs[0]:y_idxs[-1]+1])
    vvel_sinmod = np.ma.array(vvel_sinmod[x_idxs[0]:x_idxs[-1]+1, y_idxs[0]:y_idxs[-1]+1])

    # Create cKDTree object to represent the full flattened cartesian coordinates of the satellite data grid
    tree = cKDTree(list(zip(x_sat.flatten(), y_sat.flatten())))

    # Find indices of the satellite data nearest to the SINMOD coordinates we use
    x_sinmod_2d, y_sinmod_2d = np.meshgrid(x_sinmod, y_sinmod)
    d, inds = tree.query(list(zip(x_sinmod_2d.ravel(), y_sinmod_2d.ravel())), k = 1)

    # Extract satellite data by query indexes to get same grid axes and apply the same landmass mask
    sinmod_mask = np.ma.getmask(ssl_sinmod)
    sst_sat = np.ma.masked_where(sinmod_mask, sst_sat.flatten()[inds].reshape(uvel_sinmod.T.shape).T)
    ssl_sat = np.ma.masked_where(sinmod_mask, ssl_sat.flatten()[inds].reshape(uvel_sinmod.T.shape).T)
    sal_sat = np.ma.masked_where(sinmod_mask, sal_sat.flatten()[inds].reshape(uvel_sinmod.T.shape).T)
    uvel_sat = np.ma.masked_where(sinmod_mask, uvel_sat.flatten()[inds].reshape(uvel_sinmod.T.shape).T)
    vvel_sat = np.ma.masked_where(sinmod_mask, vvel_sat.flatten()[inds].reshape(uvel_sinmod.T.shape).T)

    # Rotate velocities to fit the shifted pole stereographic projection
    uvel_sinmod, vvel_sinmod = rotate_vector(uvel_sinmod, vvel_sinmod, lons_sinmod, 58)
    uvel_sat, vvel_sat = rotate_vector(uvel_sat, vvel_sat, lons_sinmod, 58)

    satellite_data = [lons_sinmod, lats_sinmod, x_sinmod, y_sinmod, ssl_sat, uvel_sat, vvel_sat]
    sinmod_data = [lons_sinmod, lats_sinmod, x_sinmod, y_sinmod, ssl_sinmod, uvel_sinmod, vvel_sinmod]
    
    return satellite_data, sinmod_data

def compare_grids(x_bnds, y_bnds,
                        img_store_fpath='eddypred',
                        sinmod_fpath='D:/master/data/compare/samples_2017.04.27_nonoverlap.nc', 
                        sat_fpath='D:/master/data/compare/phys_2016_001.nc',
                        model_fpath = 'D:/master/models/2016/cnn_mult_velocities_9652.h5'):

    """ Compare a specific area that exists within both the 
    CMEMS grid (lon:-60,60, lat:45-90) and the
    SINMOD grid (xc?, yx?)
    Specify x-y boundaries on a polar stereographic grid specified by the SINMOD projection.
    All this function does is create and store an image of both satellite and sinmod 
    ata for a specific grid. """

    fig, ax = plt.subplots(1,2,figsize=(12,6), dpi=150)
    fig.subplots_adjust(wspace=0.01, hspace=0)

    # Load both sinmod and satellite data
    satellite_data, sinmod_data = load_grid(x_bnds, y_bnds, sinmod_fpath, sat_fpath, model_fpath)

    # Get predicted satellite eddies
    cnn_win_sizes = [((int(8), int(6)), 2, 2),((int(11), int(8)), 3, 2),((int(14), int(11)), 3, 3)]
    cyc_census_df_sat, acyc_census_df_sat, = get_predicted_boxes(satellite_data, cnn_win_sizes, model_fpath, problim=0.92, storedir="sat1", plotaxis=ax[0], plot_title='Satellite')

    # Get predicted SINMOD eddies, most the same, except a more relaxed probability limit for prediction
    cnn_win_sizes = [((int(8), int(6)), 2, 2),((int(11), int(8)), 3, 2),((int(14), int(11)), 3, 3)]
    cyc_census_df_sinmod, acyc_census_df_sinmod = get_predicted_boxes(sinmod_data, cnn_win_sizes, model_fpath, problim=0.92, storedir="sinmod1", plotaxis=ax[1], plot_title='SINMOD')

    fig.savefig(img_store_fpath, bbox_inches='tight')
    
    plt.close(fig)

    return cyc_census_df_sat, acyc_census_df_sat, cyc_census_df_sinmod, acyc_census_df_sinmod

def compare_specific_grid(x_bnds, y_bnds,
                        img_store_fpath='eddypred',
                        sinmod_fpath='D:/master/data/compare/samples_2017.04.27_nonoverlap.nc', 
                        sat_fpath='D:/master/data/compare/phys_2016_001.nc',
                        model_fpath = 'D:/master/models/2016/cnn_mult_velocities_9652.h5'):

    """ Compare a specific area that exists within both the 
    CMEMS grid (lon:-60,60, lat:45-90) and the
    SINMOD grid (xc?, yx?)
    Specify x-y boundaries on a polar stereographic grid specified by the SINMOD projection.
    All this function does is create and store an image of both satellite and sinmod 
    ata for a specific grid. """

    fig, ax = plt.subplots(1,2,figsize=(12,6), dpi=150)
    fig.subplots_adjust(wspace=0.01, hspace=0)

    # Load both sinmod and satellite data
    satellite_data, sinmod_data = load_grid(x_bnds, y_bnds, sinmod_fpath, sat_fpath, model_fpath)

    # Get predicted satellite eddies
    cnn_win_sizes = [((int(8), int(6)), 2, 2),((int(11), int(8)), 3, 2),((int(14), int(11)), 3, 3)]
    cyc_census_df_sat, acyc_census_df_sat, = get_predicted_boxes(satellite_data, cnn_win_sizes, model_fpath, problim=0.92, storedir="sat1", plotaxis=ax[0], plot_title='Satellite')

    # Get predicted SINMOD eddies, most the same, except a more relaxed probability limit for prediction
    cnn_win_sizes = [((int(8), int(6)), 2, 2),((int(11), int(8)), 3, 2),((int(14), int(11)), 3, 3)]
    cyc_census_df_sinmod, acyc_census_df_sinmod = get_predicted_boxes(sinmod_data, cnn_win_sizes, model_fpath, problim=0.92, storedir="sinmod1", plotaxis=ax[1], plot_title='SINMOD')

    fig.savefig(img_store_fpath, bbox_inches='tight')
    
    plt.close(fig)

    return cyc_census_df_sat, acyc_census_df_sat, cyc_census_df_sinmod, acyc_census_df_sinmod

def get_predicted_boxes(data, win_sizes, model_fpath, problim=0.95, storedir="sat1", plotaxis=None, plot_title='Satellite'):
    """"""
    lons, lats, x, y, ssl, uvel, vvel  = data

    cyc, acyc = predict_grid(data, win_sizes, model_fpath, problim=0.95, storedir="sat1")
    nCyc, nAcyc = len(cyc), len(acyc)

    # Create mask for OW values above threshold, one that contains cyclones (negative vorticity) 
    # and vice versa for anti-cyclones
    # Default Okubo Weiss value was -1, want to use -8 to be absolutely sure, we have a more loose threshold to
    # indicate cells with vortex characteristics
    OW_start = -0.05
    OW,vorticity,OW_mask,OW_cyc_mask,OW_acyc_mask = calc_OW(lons,lats,uvel,vvel,OW_start)

    # We label all unique mask clusters
    OW_mask_labeled = measure.label(OW_mask)
    cyc_mask_labeled = measure.label(OW_cyc_mask)
    acyc_mask_labeled = measure.label(OW_acyc_mask)

    # Make the predicted boxes encompass the full cyclone or anti-cyclone clusters and return info about the eddy
    cyc_ctrIdxs, cyc_minOW, cyc_new= investigate_cluster(cyc, OW, cyc_mask_labeled, 'cyclone')
    acyc_ctrIdxs, acyc_minOW, acyc_new = investigate_cluster(acyc, OW, acyc_mask_labeled, 'anti-cyclone')

    # Eddy census dataframe to hold information about each eddy
    cyc_census_df = set_eddy_census(cyc_mask_labeled, cyc_ctrIdxs, cyc_new, cyc_minOW, lons, lats, x, y, uvel, vvel, meastype='SINMOD')
    acyc_census_df = set_eddy_census(acyc_mask_labeled, acyc_ctrIdxs, acyc_new, acyc_minOW, lons, lats, x, y, uvel, vvel, meastype='SINMOD')

    # Eddy boxes given in plot-able coordiantes following axis values
    cyc_plotcoords = box2coords(x, y, cyc_new)
    acyc_plotcoords = box2coords(x, y, acyc_new)

    if plotaxis is not None:
        plotaxis.pcolormesh(x, y, ssl.T, cmap='rainbow')
        plotaxis.streamplot(x, y, uvel.T, vvel.T, density=8) 
        plot_eddies(plotaxis, cyc_plotcoords, 'r', numbered=False) 
        plot_eddies(plotaxis, acyc_plotcoords, 'b', numbered=False) 
        plotaxis.axes.get_xaxis().set_visible(False)
        plotaxis.axes.get_yaxis().set_visible(False)
        plotaxis.set_xlim([x[0], x[-1]])
        plotaxis.set_ylim([y[0], y[-1]])
        plotaxis.set_title(plot_title)

    # For report
    fig2, ax = plt.subplots(1,1,figsize=(15,12))#, dpi=500)
    ax.pcolormesh(x, y, ssl.T, cmap='rainbow')
    ax.streamplot(x, y, uvel.T, vvel.T, density=8) 
    plot_eddies(ax, cyc_plotcoords, 'r', numbered=False) 
    plot_eddies(ax, acyc_plotcoords, 'b', numbered=False) 
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([y[0], y[-1]])

    fig2.savefig(storedir + '.png', bbox_inches='tight')

    return cyc_census_df, acyc_census_df


def predict_grid(data, win_sizes, model_fpath, problim=0.95, storedir="sat1"):

    lons, lats, x, y, ssl, uvel, vvel  = data
    # Get the predicted cyclone and anti-cyclone windows
    cyc, acyc = cnn_predict_grid(data, win_sizes, problim, model_fpath, storedir)

    # Group all predicted boxes according to overlap and number of rectangles in a group
    cyc, _ = cv2.groupRectangles(rectList=cyc, groupThreshold=2, eps=0.1)
    acyc, _ = cv2.groupRectangles(rectList=acyc, groupThreshold=2, eps=0.1)
    
    return cyc, acyc


def multi_grid_analysis_model(x_bnds, y_bnds,
                    sat_dir='D:/master/data/compare/satellite/', 
                    sinmod_dir='D:/master/data/compare/sinmod/'):
    
    compare_files = zip( os.listdir(sat_dir), os.listdir(sinmod_dir) )

    # Need to define start date to keep track of seasons etc
    startdate = dt.date(2016, 9, 10)

    cyc_census_sat_out = pd.DataFrame()
    acyc_census_sat_out = pd.DataFrame()
    cyc_census_sinmod_out = pd.DataFrame()
    acyc_census_sinmod_out = pd.DataFrame()
    out_dfs = [cyc_census_sat_out, acyc_census_sat_out, cyc_census_sinmod_out, acyc_census_sinmod_out]

    csvdirpath = 'D:/master/data/csv/'
    csvnames = ['cycSat', 'acycSat', 'cycSinmod', 'acycSinmod']

    prevDfDay = 0
    for csvId, csv in enumerate(csvnames):
        if os.path.isfile(csvdirpath + csv + '.csv'):
            out_dfs[csvId] = pd.read_csv(csvdirpath + csv + '.csv', sep='\t') 
            prevDfDay = out_dfs[csvId]['day'].iloc[-1]

    for imId, (sat_fname, simmod_fname) in enumerate(compare_files):
        print(f"\nAnalyzing day {imId+1}")

        if imId < prevDfDay: continue
        #if imId < 228 or imId > 232: continue

        #if imId<19: continue
        img_store_fpath = f'C:/Users/47415/Master/images/compare/multi/compare_image_{imId}'
        cyc_census_sat, acyc_census_sat, cyc_census_sinmod, acyc_census_sinmod = compare_grids(
                            x_bnds, y_bnds,
                            img_store_fpath=img_store_fpath, 
                            sinmod_fpath=sinmod_dir+simmod_fname, 
                            sat_fpath=sat_dir+sat_fname)

        date = startdate + dt.timedelta(days=imId)

        current_dfs = [cyc_census_sat, acyc_census_sat, cyc_census_sinmod, acyc_census_sinmod]
        for dfId, (df, name) in enumerate(zip(current_dfs, csvnames)):
            nEddies = len(df)
            df.insert(0, "date", np.full((nEddies), date))
            df.insert(0, "day", np.full((nEddies), imId+1))

            out_dfs[dfId] = out_dfs[dfId].append(df)

            # Remove all unnamed columns 
            out_dfs[dfId] = out_dfs[dfId].loc[:, ~out_dfs[dfId].columns.str.contains('^Unnamed')]

            out_dfs[dfId].to_csv(csvdirpath + name + '.csv', sep='\t', encoding='utf-8')
            

def multi_grid_analysis_insitu(x_bnds, y_bnds, data_dir='D:/master/data/compare/satellite2/'):
    

    # Need to define start date to keep track of seasons etc
    startdate = dt.date(2016, 9, 10)

    out_dfs = [pd.DataFrame(), pd.DataFrame()]

    csvdirpath = 'D:/master/data/csv/'
    csvnames = ['cycInsitu', 'acycInsitu']

    # Get SINMOD coordinate data
    x,y,_,lons,lats,sst_sinmod,_,_,_,_ =  load_nc_sinmod('D:/master/data/compare/sinmod/samples_2017.06.01_nonoverlap.nc')

    x_idxs = np.where((x >= x_bnds[0]) & (x <= x_bnds[1]))[0]
    y_idxs = np.where((y >= y_bnds[0]) & (y <= y_bnds[1]))[0]
    x = np.ma.array(x[x_idxs])
    y = np.ma.array(y[y_idxs])
    lons = np.ma.array(lons[x_idxs[0]:x_idxs[-1]+1, y_idxs[0]:y_idxs[-1]+1])
    lats = np.ma.array(lats[x_idxs[0]:x_idxs[-1]+1, y_idxs[0]:y_idxs[-1]+1])
    sst_sinmod = np.ma.array(sst_sinmod[x_idxs[0]:x_idxs[-1]+1, y_idxs[0]:y_idxs[-1]+1])
    sinmod_mask = np.ma.getmask(sst_sinmod)
    sinmod_shape = sst_sinmod.T.shape

    prevDfDay = 0
    for csvId, csv in enumerate(csvnames):
        if os.path.isfile(csvdirpath + csv + '.csv'):
            out_dfs[csvId] = pd.read_csv(csvdirpath + csv + '.csv', sep='\t') 
            prevDfDay = out_dfs[csvId]['day'].iloc[-1]

    for imId, insitu_fname in enumerate(os.listdir(data_dir)):

        day = 1+imId*7

        print(f"\nAnalyzing day {day}")
        if day < prevDfDay: continue
        #if imId > 6: break

        # Load insitu netcdf data
        lons_insitu,lats_insitu,sst_insitu,ssl_insitu,sal_insitu,uvel_insitu,vvel_insitu,t_insitu =  load_nc_insitu(data_dir+insitu_fname)

        # Get a polar steregraphic projection with cartesian irregular curvilinear grid for satellite data 
        x_insitu_2d, y_insitu_2d = grid_conversion.bl2xy(*np.meshgrid(lons_insitu, lats_insitu), FE=3254000, FN=2560000, xy_res=1, slon=58, SP=60)

        # Interpolate data and the curvilienar grid axes such that it has a higher resolution than SINMOD
        #sst_insitu = interp2d_masked(sst_insitu, lons_insitu, lats_insitu, 8, 'linear')
        ssl_insitu = interp2d_masked(ssl_insitu, lons_insitu, lats_insitu, 8, 'linear')
        #sal_insitu = interp2d_masked(sal_insitu, lons_insitu, lats_insitu, 8, 'linear')
        uvel_insitu = interp2d_masked(uvel_insitu, lons_insitu, lats_insitu, 8, 'linear')
        vvel_insitu = interp2d_masked(vvel_insitu, lons_insitu, lats_insitu, 8, 'linear')
        x_insitu = interp2d_masked(x_insitu_2d.T, lons_insitu, lats_insitu, 8, 'linear')
        y_insitu = interp2d_masked(y_insitu_2d.T, lons_insitu, lats_insitu, 8, 'linear')

        # Create cKDTree object to represent the full flattened cartesian coordinates of the satellite data grid
        tree = cKDTree(list(zip(x_insitu.flatten(), y_insitu.flatten())))

        # Find indices of the satellite data nearest to the SINMOD coordinates we use
        x_2d, y_2d = np.meshgrid(x, y)
        d, inds = tree.query(list(zip(x_2d.ravel(), y_2d.ravel())), k = 1)

        # Extract satellite data by query indexes to get same grid axes and apply the same landmass mask
        #sst = np.ma.masked_where(sinmod_mask, sst_insitu.flatten()[inds].reshape(sinmod_shape).T)
        ssl = np.ma.masked_where(sinmod_mask, ssl_insitu.flatten()[inds].reshape(sinmod_shape).T)
        #sal = np.ma.masked_where(sinmod_mask, sal_insitu.flatten()[inds].reshape(sinmod_shape).T)
        uvel = np.ma.masked_where(sinmod_mask, uvel_insitu.flatten()[inds].reshape(sinmod_shape).T)
        vvel = np.ma.masked_where(sinmod_mask, vvel_insitu.flatten()[inds].reshape(sinmod_shape).T)

        # Rotate velocities to fit the shifted pole stereographic projection
        uvel, vvel = rotate_vector(uvel, vvel, lons, 58)

        insitu_data = [lons, lats, x, y, ssl, uvel, vvel]

        win_sizes = [((int(7), int(5)), 2, 2),((int(9), int(7)), 2, 2),((int(12), int(9)), 3, 3), ((int(16), int(12)), 4, 4)]
        model_fpath = 'D:/master/models/2016/cnn_mult_velocities_9652.h5'
        #model_fpath = 'D:/master/models/best_model_975.h5'
        cyc, acyc = cnn.cnn_predict_grid(insitu_data, win_sizes, 0.92, model_fpath, storedir='insitu1')

        # Group all predicted boxes according to overlap and number of rectangles in a group
        cyc, _ = cv2.groupRectangles(rectList=cyc, groupThreshold=2, eps=0.1)
        acyc, _ = cv2.groupRectangles(rectList=acyc, groupThreshold=2, eps=0.1)

        OW_start = -0.05
        OW,vorticity,OW_mask,OW_cyc_mask,OW_acyc_mask = calc_OW(lons,lats,uvel,vvel,OW_start)

        # We label all unique mask clusters
        OW_mask_labeled = measure.label(OW_mask)
        cyc_mask_labeled = measure.label(OW_cyc_mask)
        acyc_mask_labeled = measure.label(OW_acyc_mask)

        # Make the predicted boxes encompass the full cyclone or anti-cyclone clusters and return info about the eddy
        cyc_ctrIdxs, cyc_minOW, cyc_new= investigate_cluster(cyc, OW, cyc_mask_labeled, 'cyclone')
        acyc_ctrIdxs, acyc_minOW, acyc_new = investigate_cluster(acyc, OW, acyc_mask_labeled, 'anti-cyclone')

        # Eddy census dataframe to hold information about each eddy
        cyc_census_df = set_eddy_census(cyc_mask_labeled, cyc_ctrIdxs, cyc_new, cyc_minOW, lons, lats, x, y, uvel, vvel, meastype='SINMOD')
        acyc_census_df = set_eddy_census(acyc_mask_labeled, acyc_ctrIdxs, acyc_new, acyc_minOW, lons, lats, x, y, uvel, vvel, meastype='SINMOD')

        cyc_plotcoords = box2coords(x, y, cyc_new)
        acyc_plotcoords = box2coords(x, y, acyc_new)

        img_store_fpath = f'C:/Users/47415/Master/images/compare/multi/insitu/insitu_{day-1}'
        fig, ax = plt.subplots(1,1,figsize=(10,6), dpi=150)

        ax.pcolormesh(x, y, ssl.T, cmap='rainbow')
        ax.streamplot(x, y, uvel.T, vvel.T, density=8) 
        plot_eddies(ax, cyc_plotcoords, 'r', numbered=False) 
        plot_eddies(ax, acyc_plotcoords, 'b', numbered=False) 
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_xlim([x[0], x[-1]])
        ax.set_ylim([y[0], y[-1]])
        ax.set_title('In-situ')

        fig.savefig(img_store_fpath, bbox_inches='tight')

        date = startdate + dt.timedelta(days=day-1)

        print(cyc_census_df)
        print()
        print(acyc_census_df)

        current_dfs = [cyc_census_df, acyc_census_df]
        for dfId, (df, name) in enumerate(zip(current_dfs, csvnames)):
            nEddies = len(df)
            df.insert(0, "date", np.full((nEddies), date))
            df.insert(0, "day", np.full((nEddies), day))

            out_dfs[dfId] = out_dfs[dfId].append(df)

            # Remove all unnamed columns 
            out_dfs[dfId] = out_dfs[dfId].loc[:, ~out_dfs[dfId].columns.str.contains('^Unnamed')]

            out_dfs[dfId].to_csv(csvdirpath + name + '.csv', sep='\t', encoding='utf-8')
            
    

if __name__ == '__main__':
    
    
    # Choose a SINMOD Grid which will be the basis of analysis
    # Norskekysten mellom Bergen og Bodø
    #x_bnds = [720000, 1560000]
    #y_bnds = [650000, 1200000]

    # Mellom Grønland, Svalbard og Norge
    x_bnds = [1800000, 2150000]
    y_bnds = [1600000, 1870000]
    #x_bnds = [1900000, 2100000]
    #y_bnds = [1700000, 1850000]

    # Et sted i Barentshavet
    #x_bnds = [2400000, 2900000]
    #y_bnds = [900000, 1200000]

    # Skagerak
    #x_bnds = [400000, 880000]
    #y_bnds = [150000, 450000]

    # Mellom Svalbard og Grønland, potensielt mye rart som slipper mellom der
    #x_bnds = [2200000, 2800000]
    #y_bnds = [1750000, 2350000]
    
    # Full mellom norge og Grønland
    x_bnds = [1000000, 2000000]
    y_bnds = [1000000, 2100000]
    
    
    #multi_grid_analysis_model(x_bnds, y_bnds)
    multi_grid_analysis_insitu(x_bnds, y_bnds)



