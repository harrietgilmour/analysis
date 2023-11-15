# Functions for the main program

# Import the relevant modules
import iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import iris.plot as iplt
import cartopy.crs as ccrs
import datetime
import shutil
from six.moves import urllib
from pathlib import Path
from math import pi
import trackpy
from iris.time import PartialDateTime
import tobac
import warnings
import math

# First set up the function for constraining OLR
def load_olr_data(data_file):
    """
    Loads in CPM 1 hourly OLR data and sets it up for use.
    
    Parameters:
    data_file (str): The path to the OLR data file.
    
    Returns:
    iris.cube.Cube: The OLR data.
    iris.coords.DimCoord: The time coordinate of the OLR data.
    list: The datetimes of the OLR data.
    iris.Constraint: The constraint for the first time step of the OLR data.
    """
    olr = iris.load_cube(data_file)

    #constraining the dataset by time so it runs quicker:
    #week = iris.Constraint(time=lambda cell: cell.point.day <= 7)
    #olr = olr.extract(week)
    olr.coord('time').bounds = None #REMOVING BOUNDS FROM TIME TO SEE IF THIS HELPS THE TYPEERROR

    # Remove coord system or else the animations don't run (suggested by AVD team)
    olr.coord('latitude').coord_system = None
    olr.coord('longitude').coord_system = None

    # Code from the AVD team
    time = olr.coord('time')
    datetimes = time.units.num2date(time.points)
    con = iris.Constraint(time=datetimes[0])
    olr.extract(con)


    return olr, time, datetimes, con

# Function to set up the directory to save outputs and plots
def setup_directories():
    """
    Sets up directories to save output and plots.
    """
    savedir = Path("Save")
    if not savedir.is_dir():
        savedir.mkdir()
    plot_dir = Path("Plot")
    if not plot_dir.is_dir():
        plot_dir.mkdir()

    return savedir, plot_dir


# Proposed wrapper function for calculating dxy
# Given the longitude, latitude and olr data, calculate the spatial and temporal resolution of the input data.
def calculate_dxy(longitude, latitude, olr):
    """
    Calculates the spatial and temporal resolution of the input data.
    
    Parameters:
    longitude (numpy.ndarray): The longitudes of the input data.
    latitude (numpy.ndarray): The latitudes of the input data.
    olr (iris.cube.Cube): The input data.
    
    Returns:
    float: The spatial resolution of the input data.
    float: The temporal resolution of the input data.
    """
    R = 6.3781e6
    dx = np.gradient(longitude)[1]
    dx = dx * (pi / 180) * R * np.cos(latitude * pi / 180)
    dy = np.gradient(latitude)[0]
    dy = dy * (pi / 180) * R
    dxy, dt = tobac.get_spacings(olr, grid_spacing=4500, time_spacing=3600)
    
    return dxy, dt

# Function for calculating the brightness temperatures from the OLR data
def calculate_temperatures(a, b, sigma, olr):
    """
    Calculates the temperatures of the input data.
    
    Parameters:
    a (float): The value of a.
    b (float): The value of b.
    sigma (float): The value of sigma.
    olr (iris.cube.Cube): The input data.
    
    Returns:
    iris.cube.Cube: The temperatures of the input data.
    iris.cube.Cube: The tb_var of the input data.
    iris.cube.Cube: The tb of the input data.
    """
    tf = (olr.data/sigma)**(0.25)
    tb_var = (-a + np.sqrt(a**2 + 4*b*tf.data))/(2*b)
    
    tb = olr.copy()
    tb.data = tb_var.data

    return tf, tb_var, tb

# Function for loading the precip data
# Very similar to loading olr data
# But doesn't modify precip.coord system
def load_precip_data(precip_path):
    """
    Loads in precipitation data and sets it up for use.
    
    Parameters:
    precip_path (str): The path to the precipitation data file.
    
    Returns:
    iris.cube.Cube: The precipitation data.
    iris.coords.DimCoord: The time coordinate of the precipitation data.
    list: The datetimes of the precipitation data.
    iris.Constraint: The constraint for the first time step of the precipitation data.
    """
    precip = iris.load_cube(precip_path)
    week = iris.Constraint(time=lambda cell: cell.point.day <= 31)
    precip = precip.extract(week)
    precip.coord('time').bounds = None
    time = precip.coord('time')
    datetimes = time.units.num2date(time.points)
    con = iris.Constraint(time=datetimes[0])
    precip.extract(con)
    return precip, time, datetimes, con

# Function for setting up parameters_features
def setup_parameters_features(position_threshold, sigma_threshold, target, threshold, n_min_threshold):
    """
    Sets up the parameters for feature detection.
    
    Parameters:
    position_threshold (str): The position threshold.
    sigma_threshold (float): The sigma threshold.
    target (str): The target.
    threshold (list): The threshold.
    n_min_threshold (int): The minimum number of threshold.
    
    Returns:
    dict: The parameters for feature detection.
    """
    parameters_features = {}
    parameters_features['position_threshold'] = position_threshold
    parameters_features['sigma_threshold'] = sigma_threshold
    parameters_features['target'] = target
    parameters_features['threshold'] = threshold
    parameters_features['n_min_threshold'] = n_min_threshold
    return parameters_features

# Function for setting up parameters segmentation
def setup_parameters_segmentation(target, method, threshold):
    """
    Sets up the parameters for segmentation.
    
    Parameters:
    target (str): The target.
    method (str): The method.
    threshold (int): The threshold.
    
    Returns:
    dict: The parameters for segmentation.
    """
    parameters_segmentation = {}
    parameters_segmentation['target'] = target
    parameters_segmentation['method'] = method
    parameters_segmentation['threshold'] = threshold
    return parameters_segmentation

# Function for setting up parameters linking
def setup_parameters_linking(v_max, stubs, order, extrapolate, memory, adaptive_stop, adaptive_step, subnetwork_size, method_linking):
    """
    Sets up the parameters for linking.
    
    Parameters:
    v_max (int): The maximum velocity.
    stubs (int): The minimum number of timesteps for a tracked cell to be reported.
    order (int): The order.
    extrapolate (int): The extrapolation.
    memory (int): The memory.
    adaptive_stop (float): The adaptive stop.
    adaptive_step (float): The adaptive step.
    subnetwork_size (int): The subnetwork size.
    method_linking (str): The method for linking.
    
    Returns:
    dict: The parameters for linking.
    """
    parameters_linking = {}
    parameters_linking['v_max'] = 60
    parameters_linking['stubs'] = stubs
    parameters_linking['order'] = order
    parameters_linking['extrapolate'] = extrapolate
    parameters_linking['memory'] = memory
    parameters_linking['adaptive_stop'] = adaptive_stop
    parameters_linking['adaptive_step'] = adaptive_step
    parameters_linking['subnetwork_size'] = subnetwork_size
    parameters_linking['method_linking'] = method_linking
    return parameters_linking

# Function which performs the feature detection
def perform_feature_detection(tb, dxy, savedir, parameters_features):
    """
    Performs feature detection on input data and saves results to file.
    
    Parameters:
    tb (iris.cube.Cube): The input data.
    dxy (tuple): The spatial resolution of the input data.
    savedir (str): The directory to save the output files.
    parameters_features (dict): The parameters for feature detection.
    
    Returns:
    tobac.utils.FeatureDetection: The features detected in the input data.
    """
    savedir = Path(savedir)
    if not savedir.is_dir():
        savedir.mkdir()
    
    # Feature detection and save results to file:
    print('starting feature detection')
    Features = tobac.feature_detection_multithreshold(tb, dxy, **parameters_features)
    Features.to_hdf(savedir / 'Features.h5', 'table')
    print('feature detection performed and saved')
    
    return Features

# Function which performs the segmentation
def perform_segmentation(tb, dxy, savedir, parameters_segmentation, Features):
    """
    Performs segmentation on input data and saves results to file.
    
    Parameters:
    tb (iris.cube.Cube): The input data.
    dxy (tuple): The spatial resolution of the input data.
    savedir (str): The directory to save the output files.
    parameters_segmentation (dict): The parameters for segmentation.
    Features (tobac.utils.FeatureDetection): The features detected in the input data.
    
    Returns:
    tuple: The mask and features segmented from the input data.
    """
    
    # Perform segmentation and save results to files:
    Mask_tb, Features_tb = tobac.segmentation_2D(Features, tb, dxy, **parameters_segmentation)
    print('segmentation tb performed, start saving results to files')
    iris.save([Mask_tb], savedir / 'Mask_Segmentation_tb.nc', zlib=True, complevel=4)
    Features_tb.to_hdf(savedir / 'Features_tb.h5', 'table')
    print('segmentation tb performed and saved')
    
    return Mask_tb, Features_tb

# Function which performs the linking
# and saves the results to file
def perform_linking(Features, tb, dt, dxy, savedir, parameters_linking):
    """
    Performs linking on input data and saves results to file.
    
    Parameters:
    Features (tobac.utils.FeatureDetection): The features detected in the input data.
    tb (iris.cube.Cube): The input data.
    dt (float): The time resolution of the input data.
    dxy (tuple): The spatial resolution of the input data.
    savedir (str): The directory to save the output files.
    parameters_linking (dict): The parameters for linking.
    
    Returns:
    pandas.DataFrame: The tracks detected in the input data.
    """

    # Perform linking and save results to file:
    Track = tobac.linking_trackpy(Features, tb, dt=dt, dxy=dxy, **parameters_linking)
    Track["longitude"] = Track["longitude"] - 360
    Track.to_hdf(savedir / 'Track.h5', 'table')
    print('linking performed and saved')
    
    return Track

# Function which performs the analysis
def perform_analysis(Features, Features_tb, Mask_tb, Track, parameters_features):
    """
    Performs analysis on input data and returns results as a dictionary.
    
    Parameters:
    Features (pandas.DataFrame): The features detected in the input data.
    Features_tb (pandas.DataFrame): The area surrounding the detected feature.
    Mask_tb (iris.cube.Cube): The mask segmented from the input data.
    Track (pandas.DataFrame): The tracks detected in the input data.
    parameters_features (dict): The parameters for feature detection.
    
    Returns:
    dict: A dictionary containing the results of the analysis.
    """

    results = {}
    
    # Number of features detected:
    CC_Features = Features[Features['threshold_value'] < (parameters_features['threshold'] + 1)]
    results['num_features'] = CC_Features.count()[0]
    
    # Average size of segmented areas associated with feature in a track:
    a = []
    for cell in np.unique(Track.cell.values):
        gcells= np.mean(np.array(Track[Track.cell== cell].num.values))
        area=gcells*(4.5**2)
        a.append(area)
    a = np.array(a)
    mean_a = a.mean()
    results['mean_area'] = mean_a
    
    #area = tobac.analysis.calculate_area(Features, Mask_tb, method_area='latlon')
    #mean_area = area.mean()
    #results['mean_area_tobac_analysis'] = mean_area
    
    # Max size of segmented areas associated with feature:
    max_a = a.max()
    results['max_area'] = max_a

    # Min size of segmented areas associated with feature:
    min_a = a.min()
    results['min_area'] = min_a
    
    # Number of tracks detected:
    results['num_tracks'] = len(Track['cell'].dropna().unique()) - 1
    
    # Average lifetime of tracks:
    cell = Track.groupby("cell")
    minutes = (cell["time_cell"].max() / pd.Timedelta(minutes=1)).values
    lifetime = minutes/60 #converting from minutes to hours
    lifetime_hrs = lifetime[1:] #removes the first large value from the mean and max calculations
    lifetime_mean = lifetime_hrs.mean()
    results['mean_lifetime'] = lifetime_mean
    
    # Max lifetime of tracks:
    lifetime_max = lifetime_hrs.max()
    results['max_lifetime'] = lifetime_max
    
    # Min lifetime of tracks:
    lifetime_min = lifetime_hrs.min()
    results['min_lifetime'] = lifetime_min
    
    # Mean velocity of MCSs:
    vel = tobac.analysis.calculate_velocity(Track, method_distance=None)
    v = []
    for cell in np.unique(vel.cell.values):
        vel.replace([np.inf, -np.inf], np.nan, inplace=True)
        ps = np.nanmean(vel[vel.cell== cell].v.values)
        v.append(ps)
    v = np.array(v)
    v = v[v<math.inf]
    v = v[1:] #removes the one ridiculously large value at the start of each array
    print(v)
    results['mean_velocity'] = v.mean()
    
    # Max velocity of MCSs:
    results['max_velocity'] = v.max()
    
    # Min velocity of MCSs:
    results['min_velocity'] = v.min()
    
    return results

# Function which performs the sensitivity analysis
def perform_sensitivity_analysis(tb, savedir, parameters_features, parameters_segmentation, parameters_linking, threshold_values):
    """
    Performs sensitivity analysis for different values of parameters_features['threshold'] and parameters_segmentation['threshold'].
    
    Parameters:
    tb (iris.cube.Cube): The input data.
    savedir (str): The directory to save the output files.
    parameters_features (dict): The parameters for feature detection.
    parameters_segmentation (dict): The parameters for segmentation.
    parameters_linking (dict): The parameters for linking.
    threshold_values (list): A list of threshold values to use for sensitivity analysis.
    
    Returns:
    pandas.DataFrame: A DataFrame containing the results of the analysis for each value of 'threshold'.
    """

    savedir = Path(savedir)
    if not savedir.is_dir():
        savedir.mkdir()
    
    
    results = []
    for threshold in threshold_values:
        # Set up parameters:
        #parameters_linking['vmax'] = threshold
        # Below 2 are for single threshold Tb analysis:
        #parameters_features['threshold'] = threshold
        #parameters_segmentation['threshold'] = threshold
        #Below is for n_min_threshold:
        parameters_features['threshold'] = threshold
        parameters_segmentation['threshold'] = threshold


        dxy, dt = tobac.get_spacings(tb, grid_spacing=4500, time_spacing=3600)
        
        # Feature detection:
        Features = tobac.feature_detection_multithreshold(tb, dxy=calculate_dxy,**parameters_features)
        
        # Segmentation:
        Mask_tb, Features_tb = tobac.segmentation_2D(Features, tb, dxy,**parameters_segmentation)
        
        # Linking:
        Track = tobac.linking_trackpy(Features, tb, dt=dt, dxy=dxy, **parameters_linking)
        Track["longitude"] = Track["longitude"] - 360
        Track.to_hdf(savedir / 'Jul_1998/singleTb/Track_{0}.h5'.format(threshold), 'table')
        
        # Analysis:
        analysis_results = perform_analysis(Features, Features_tb, Mask_tb, Track, parameters_features)
        analysis_results['threshold'] = threshold
        results.append(analysis_results)

        print('Finished analysis for threshold = {0}'.format(threshold))
        
    # Save results to file:
    results_df = pd.DataFrame(results)
    results_df.to_csv(savedir / 'Jul_1998/singleTb/sensitivity_analysis_singleTbThreshold.csv', index=False)
    
    return results_df


#function for MCS initiation time
"""
    Calculates the initiation time of MCSs
    """
def get_mcs_init(mcstracks):
    mcstracks['hour']= mcstracks.datetime.dt.hour
    diurnal=[]
    for cell in np.unique(mcstracks.cell.values):
        init_hour = mcstracks[mcstracks.cell == cell].hour.values[0]
        diurnal.append(init_hour)
    return diurnal


#function for MCS dissipation time
"""
    Calculates the dissipation time of MCSs
    """
def get_mcs_diss(mcstracks):
    mcstracks['hour']= mcstracks.datetime.dt.hour
    diurnal=[]
    for cell in np.unique(mcstracks.cell.values):
        diss_hour = mcstracks[mcstracks.cell == cell].hour.values[-1]
        diurnal.append(diss_hour)
    return diurnal


#####for above two functions, they can be followed with the following code to produce histograms of initiation and dissipation time frequency:
#init_hours = get_mcs_init(mcstracks)
#init_hours = np.array(init_hours)
#init_hours, bins = np.histogram(init_hours, bins = np.arange(0,25))

#diss_hours = get_mcs_diss(mcstracks)
#diss_hours = np.array(diss_hours)
#diss_hours, bins = np.histogram(diss_hours, bins = np.arange(0,25))

#init = init_hours/np.nansum(init_hours) * 100
#diss = diss_hours/np.nansum(diss_hours) * 100

#import seaborn as sns 
#sns.set()

#plt.plot(np.arange(0,24), init, label = 'initiation ', color= 'darkblue', linewidth = 1)
#plt.plot(np.arange(0,24), diss, label = 'dissipation ', color= 'darkblue', linewidth = 1, linestyle = 'dotted')
#plt.xticks(np.arange(0,23)[::2],fontsize=16)
#labels= ti.astype(str)
#plt.xticklabels(labels[::2],fontsize= 16)
#plt.yticks(np.arange(2,12,2))
#plt.xlabel('Time (hour of day)',fontsize=16)
#plt.xlim(-0.5,22.5)
#plt.ylabel('Frequency [%]', fontsize = 16 )
#plt.yticks(fontsize=16)
#plt.legend(loc='best',fontsize=12)

# function for MCS peak maturity time (based on maximum precip)
def get_mcs_max_precip(mcstracks): 
    mcstracks['hour'] = mcstracks.datetime.dt.hour
    rain_peak=[]
    for cell in np.unique(mcstracks.cell.values):
        subset = mcstracks[mcstracks.cell == cell]
        peak = np.nanmax(subset.total_precip.values)
        hour = subset[subset.total_precip == peak].hour.values[0]
        rain_peak.append(hour)
    rain_histo = np.histogram(rain_peak, bins = np.arange(0,25))
    return rain_histo[0]


# function for MCS peak maturity time (based on minimum Tb)
def get_mcs_min_tb(mcstracks): 
    mcstracks['hour'] = mcstracks.datetime.dt.hour
    tb_peak=[]
    for cell in np.unique(mcstracks.cell.values):
        subset = mcstracks[mcstracks.cell == cell]
        peak = np.nanmin(subset.tb_min.values)
        hour = subset[subset.tb_min == peak].hour.values[0]
        tb_peak.append(hour)
    tb_histo = np.histogram(tb_peak, bins = np.arange(0,25))
    return tb_histo[0]

# function for MCS peak maturity time (based on maximum updraft velocity)
def get_mcs_max_w(mcstracks): 
    mcstracks['hour'] = mcstracks.datetime.dt.hour
    w_peak=[]
    for cell in np.unique(mcstracks.cell.values):
        subset = mcstracks[mcstracks.cell == cell]
        peak = np.nanmax(subset.w_max.values)
        hour = subset[subset.w_max == peak].hour.values[0]
        w_peak.append(hour)
    tb_histo = np.histogram(w_peak, bins = np.arange(0,25))
    return tb_histo[0]



#function to calculate velocity/propagation speed of MCSs
"""
    Calculates a histogram for the velocity/propagation speed of MCSs

    Parameters: 
    vel (need 'v' column in Track pandas dataframe - NEED TO DO TOBAC.CALCULATE_VELOCITY FIRST TO ADD THIS IN BEFORE RUNNING THE FUNCTION)
    """

def get_v(vel):
    v= []
    for cell in np.unique(vel.cell.values):
        ps = np.nanmean(vel[vel.cell== cell].v.values)
        v.append(ps)
    v = np.array(v)
    v = v[v<math.inf] 
    print('propagation speed histo calculated.')
    return v

#can then do the following to create a the histogram:
# velocity=get_v(vel)
# bins=velocity[1]
#ticks=np.arange(bins.shape[0]- 1)
#plt.bar(ticks, velocity[0]/ np.nansum(velocity[0]) * 100 , width=0.9,color= 'teal')
#plt.xticks(fontsize = 12)
#plt.yticks(fontsize= 12)
#plt.xlabel('propagation speed (m s$^{-1}$)', fontsize= 16)
#plt.ylabel('Frequency (%)', fontsize= 16)


#function to calculate total distance travelled of MCSs
"""
    Calculates a histogram for the total distance travelled of MCSs

    Parameters: 
    dist_tracks (need 'distance' column in Track pandas dataframe - NEED TO DO TOBAC.CALCULATE_DISTANCE FIRST TO ADD THIS IN BEFORE RUNNING THE FUNCTION)
    """

def get_distance(tracks):
    total_distance=[]

    for cell in np.unique(tracks.cell.values):
        subset = tracks[tracks.cell == cell]
        subset['distance']=0
        for x in np.arange(subset.shape[0]):
            if x != np.arange(subset.shape[0])[-1]:
                dist = tobac.calculate_distance(subset.iloc[x], subset.iloc[x+1], method_distance=None)
                subset['distance'].iloc[x] = dist
                x = x+1
            elif x == np.arange(subset.shape[0])[-1]:
                subset['distance'].iloc[x] = np.nan

        total_dist = np.sum(subset['distance'])
        total_distance.append(total_dist)
    print('distance travelled array calculated for {}.'.format(str(tracks)))
    return total_distance



#function to get the locations (lats + lons) of initiation and dissipation of MCSs
"""
    Provides the spatial locations  of MCS initiation and dissipation

    Parameters: 
    mcstracks (pandas dataframe of tracks)
    """
def get_init(mcstracks):
    init_lats= []
    init_lons= []
    diss_lats= []
    diss_lons= []
    for cell in np.unique(mcstracks.cell.values):
        subset= mcstracks[mcstracks.cell == cell]
        init_lats.append(subset.latitude.values[0])
        init_lons.append(subset.longitude.values[0])
        diss_lats.append(subset.latitude.values[-1])
        diss_lons.append(subset.longitude.values[-1])
    return np.array(init_lats), np.array(init_lons), np.array(diss_lats), np.array(diss_lons)

#can then create a map of the output using:
#init_lats, init_lons, diss_lats, diss_lons = get_init(mcstracks)
#plt.figure(figsize=(40,20))
#xlabels=[-90,-70,-50,-30]
#ylabels= [-40,-30,-20,-10,0,10]
## markersize 
#s = 100


# Locations of initiation and dissipation of MCSs 
#ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
#ax1.scatter(init_lons, init_lats,  color='crimson',marker='o', s = s, transform=ccrs.PlateCarree(),label= 'initiation')
#ax1.scatter(diss_lons, diss_lats,  color='pink',marker='o', s = s/2, transform=ccrs.PlateCarree(),label= 'dissipation')

#ax1.coastlines()
#ax1.legend(fontsize= 25)
#ax1.set_xticks(xlabels, xlabels)
#ax1.set_yticks(ylabels,ylabels)
#ax1.set_xlabel('Lon $^\circ$E',  fontsize=25)
#ax1.set_ylabel('Lat $^\circ$N',  fontsize=25)


#function to get an array of MCS areas (from just features that are linked in tracks)
"""
    Provides an array of MCS areas for features that are associated with a track. 
    The function includes the conversion from grid points to area.

    Parameters: 
    mcstracks (pandas dataframe of tracks)
    """
def get_area(mcstracks):
    a= []
    for cell in np.unique(mcstracks.cell.values):
        gcells= np.mean(np.array(mcstracks[mcstracks.cell== cell].num.values))
        area=gcells*(4.5**2)
        a.append(area)
    a = np.array(a)
    print('array of MCS areas generated.')
    return a

#This can then be used to generate mea, max and min areas, as well as a histogram of the distribution:
#area=get_area(mcstracks)
#print(area)
#area.min() (or mean, max)
# plt.figure(figsize=(16,10))
#plt.hist(area,bins=[40000,80000,120000,160000,200000,240000,280000,320000,360000,400000,440000],width=30000,color= 'teal')
#plt.xlabel('area (km$^{2}$)', fontsize= 20)
#plt.ylabel('Frequency [%]', fontsize= 20)
#plt.xticks([40000,80000,120000,160000,200000,240000,280000,320000,360000,400000,440000],fontsize=18)
#plt.yticks(fontsize=18)



#function to get an array of MCS lifetimes (from just features that are linked in tracks)
"""
    Provides an array of MCS lifetimes for features that are associated with a track. 
    The function includes the conversion from grid points to area.

    Parameters: 
    mcstracks (pandas dataframe of tracks)
    """
def get_lifetime(mcstracks):
    cell = Track.groupby("cell")
    minutes = (cell["time_cell"].max() / pd.Timedelta(minutes=1)).values
    lifetime = minutes/60 #converting from minutes to hours
    lifetime_hrs = lifetime[1:] #removes the first large value from the mean and max calculations
    print('MCS lifetimes generated')
    return lifetime_hrs
    