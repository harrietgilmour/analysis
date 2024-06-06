# Python script for feature detection and segmentation of MCSs based on single brightness temp value of 240K using tobac.
# using yearly files of Tb as input and yearly features and segmentation files as output rather than monthly
#
# <USAGE> python feature_detection.py <TB_FILE>
#
# <EXAMPLE> python feature_detection.py /data/uers/hgilmour/tb/2005/tb_2005.nc
#

# Import local packages
import os
import sys
import glob

# Import third party packages
import iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import iris.plot as iplt
import datetime
import shutil
from six.moves import urllib
from pathlib import Path
import trackpy
from iris.time import PartialDateTime
import tobac

# Import and set up warnings
import warnings
warnings.filterwarnings('ignore')

# Define the usr directory for the dictionaries
sys.path.append("/project/cssp_brazil/mcs_tracking_HG")

# Import functions and dictionaries
import dictionaries as dic

# Function that will check the number of arguements passed
def check_no_args(args):
    """ Check the number of arguements passed"""
    if len(args) != 3:
        print("Incorrect number of arguements")
        print("Usage: python feature_detection.py <TB_FILE>")
        print("Example: python feature_detection.py /data/users/hgilmour/tb/2005/tb_2005.nc")
        sys.exit(1)


# Write a function which loads the file
def open_dataset(ccpf_tracks_file):
    """ Load specified files"""

    CCPF_tracks = pd.read_hdf(ccpf_tracks_file, 'table') # load all the cubes in the input file

    return CCPF_tracks

# Write a function which loads the file
def open_dataset_tb(tb_file):
    """ Load specified files"""

    tb = iris.load(tb_file) # load all the cubes in the input file
    tb = tb[0]
    return tb



# Define main function 
def main():
    #First extract the arguments:
    ccpf_tracks_file = str(sys.argv[1])
    tb_file = str(sys.argv[2])

    # We want to extract the month and year from the tb_file path
    # An example of the path is:
    # /data/users/hgilmour/tb/2005/tb_merge_01_2005.nc
    # Extract the filename first
    filename = os.path.basename(ccpf_tracks_file)
    print("Type of filename:", type(filename))
    print("Filename:", filename)
    filename_without_extension = os.path.splitext(filename)
    #print("Type of filename_without_extension:", type(filename_without_extension))
    #print(filename_without_extension)
    filename = filename.replace(".", "_")
    segments = filename.split("_")
    print(segments)
    #segments = segments.split("_")
    #print(segments)
    year = segments[1]

    # Print the year
    print("year", year)

    # check the number of arguements:
    check_no_args(sys.argv)


    # Determine temporal and spatial sampling of the input data:
    dxy = 10000 ## THIS SHOULD BE 10000 FOR REGRIDDED AND INTERPOLATED DATA
    dt = 3600

    Features = open_dataset(ccpf_tracks_file)

    tb = open_dataset_tb(tb_file)


    # Segmentation
    parameters_segmentation={}
    parameters_segmentation['target']='minimum'
    parameters_segmentation['method']='watershed'
    parameters_segmentation['threshold']=240

    segmentation_filename = f"segmentation_yearly_{year}_INTERP_CCPF.nc"
    print("segmentation filename", segmentation_filename)

    segmentation_savepath = "/project/cssp_brazil/mcs_tracking_HG/OBS_TRACKS/segmentation_obs/" + segmentation_filename

    features_tb_filename = f"features_tb_yearly_{year}_INTERP_CCPF.h5"
    print("features_tb filename", features_tb_filename)

    features_tb_savepath = "/project/cssp_brazil/mcs_tracking_HG/OBS_TRACKS/features_tb_obs/" + features_tb_filename

    # Perform segmentation and save results to files:
    Mask_tb,Features_tb=tobac.segmentation_2D(Features,tb,dxy,**parameters_segmentation)
    print('segmentation tb performed, start saving results to files')
    iris.save([Mask_tb], segmentation_savepath, zlib=True, complevel=4)
    Features_tb.to_hdf(features_tb_savepath, 'table')
    print('segmentation tb performed and saved')

#Run the main function
if __name__ == "__main__":
    main()




