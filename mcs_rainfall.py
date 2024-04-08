# Python script to generate array of unique cell values in a month of tobac tracking
#
# <USAGE> python unique_cells.py <TRACKS_FILE>
#
# <EXAMPLE> python unique_cells.py /project/cssp_brazil/mcs_tracking_HG/init_tracks_obs/tracks_2006_01.h5
#


# Import local packages
import os
import sys
import glob

# Import third party packages
import numpy as np
import pandas as pd
import xarray as xr

# Import and set up warnings
import warnings
warnings.filterwarnings('ignore')


# Write a function which will check the number of arguements passed
def check_no_args(args):
    """Check the number of arguements passed"""
    if len(args) != 4:
        print('Incorrect number of arguements')
        print('Usage: python mcs_rainfall.py <PRECIP_FILE> <MASK_FILE> <MONTH>')
        print('Example: python MCS_RAINFALL.py /scratch/hgilmour/obs/precip/precip_2001.nc' '/project/cssp_brazil/mcs_tracking_HG/OBS_TRACKS/fsegmentation_obs/segmentation_yearly_2001_INTERP_CCPF.nc' '01')
        sys.exit(1)


# Write a function which loads the file
def open_dataset(mask_file, precip_file, month):
    """Load specified files"""
    
    print("opening datasets")
    print(month)

    #Load precip and CCPF mask files
    mask_CCPF = xr.open_dataset(mask_file)
    mask_CCPF = mask_CCPF.segmentation_mask

    # subset mask to only include the month of interest
    mask_CCPF = mask_CCPF.where(mask_CCPF.time.dt.month == int(month), drop=True)

    precip = xr.open_dataset(precip_file)
    #precip = precip.precipitation_flux ## THIS ONE FOR 2001-2005

    precip = precip.precipitationCal ## THIS ONE FOR 2006-2007
    precip = precip[:,:,1:] ## THIS ONE FOR 2006-2007
    precip = precip.transpose('time', 'lat', 'lon') ## THIS ONE FOR 2006-2007
    precip = precip.rename({'lat': 'latitude', 'lon': 'longitude'}) ## THIS ONE FOR 2006-2007

    return mask_CCPF, precip

# Write a function which will check the number of arguements passed
def check_dataset_size(mask_CCPF, precip):
    """Check the number of arguements passed"""
    if (precip.shape == mask_CCPF.shape) !=True:
        print('Dataset shapes do not match')
        print('Precip shape:', precip.shape)
        print('Mask shape:', mask_CCPF.shape)
        sys.exit(1)

# Write a function that creates an empty precip dataset to later append MCS precip to
def create_empty_dataset(precip):
    nt,nx,ny = precip.shape
    subset_new = np.zeros((nt,nx,ny))

    mcs_precip = precip.copy(data=subset_new)

    return mcs_precip

# Write a function that loops through MCS masks and appends only MCS precip to the empty dataset
def find_mcs_precip(mask_CCPF, precip, mcs_precip):
    for i in np.arange(0, precip.shape[0]): ## for each timestep in the year file
        mcs_precip[i,:,:] = precip[i,:,:].where(mask_CCPF[i,:,:] > 0) ## append only the precip values where the mask is to the new xarray dataarray. This selects only precip from MCSs into the new dataarray

    return mcs_precip


#Define the main function / filerting loop:
def main():
    """Main function."""

    # First extract the arguements:
    precip_file = str(sys.argv[1])
    mask_CCPF_file = str(sys.argv[2])
    month = str(sys.argv[3])

    #check the number of arguements
    check_no_args(sys.argv)

    #find the year of the file
    filename = os.path.basename(precip_file)
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
    print("year:", year)
    month = segments[2]
    print("month:", month)  

     #first open the datasets for 1 month
    mask_CCPF, precip = open_dataset(mask_CCPF_file, precip_file, month) 
    
    print("mask_CCPF month:", mask_CCPF.time.dt.month)

    check_dataset_size(mask_CCPF, precip)

    # print the shape of the datasets
    print('mask_CCPF shape:', mask_CCPF.shape)

    mcs_precip = create_empty_dataset(precip)       

    mcs_precip = find_mcs_precip(mask_CCPF, precip, mcs_precip)

    # Save the unique cells array in the unique_cell_files directory
    mcs_precip.to_netcdf('/project/cssp_brazil/mcs_tracking_HG/OBS_TRACKS/mcs_rainfall/mcs_rainfall_OBS_{}_{}.nc'.format(year, month))

    print('Created MCS precip for month {}, and year {}'.format(month, year))



#Run the main function
if __name__ == "__main__":
    main()