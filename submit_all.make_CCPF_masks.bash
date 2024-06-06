#!/bin/sh -l
#
# This script submits the jobs to run the feature_detection_yearly.py script
#
# Usage: submit_all.feature_detection.bash <year>
#
# For example: bash submit_all.feature_detection.bash 2005
#

# Check that the year has been provided
if [ $# -ne 1 ]; then
    echo "Usage: submit_all.unique_cells.bash <year>"
    exit 1
fi

# extract the year from the command line
year=$1

# echo the year
echo "Finding unique cells for month in year: $year"

# set up the extractor script
EXTRACTOR="/data/users/hgilmour/initial_tracks/tobac_initial_tracks/submit.make_CCPF_masks.sh"

# base directory is the directory where the tb files are stored
# in format tb_merge_mm_yyyy.nc

base_dir_tb="/scratch/hgilmour/obs/tb/annual_files_hrly/regridded/interpolated"

base_dir_tracks="/project/cssp_brazil/mcs_tracking_HG/OBS_TRACKS/final_tracks_obs/merged/${year}"

# Set up the output directory
OUTPUT_DIR="/project/cssp_brazil/mcs_tracking_HG/lotus_output/CCPF_masks"
mkdir -p $OUTPUT_DIR

    
echo $year


# Find the tb files for the year
tb_file="interp_regridded_tb_${year}.nc"
# construct the tracks path
tb_path=${base_dir_tb}/${tb_file}

# Find the tracks files for the year
ccpf_tracks_file="CCPF_${year}.hdf"
# construct the tracks path
tracks_path=${base_dir_tracks}/${ccpf_tracks_file}

# Set up the output files
OUTPUT_FILE="$OUTPUT_DIR/obs.$year.out"
ERROR_FILE="$OUTPUT_DIR/obs.$year.err"

# submit the batch job
sbatch --mem=200000 --ntasks=4 --time=250 --output=$OUTPUT_FILE --error=$ERROR_FILE $EXTRACTOR $tracks_path $tb_path  ## ADD --QOS=LONG BACK IN FOR NORMAL USE, --mem=220000, --time=400

