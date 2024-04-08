#!/bin/sh -l
#
# This script submits the jobs to run the unique_cells.py script
#
# Usage: submit_all.unique_cells.bash <year>
#
# For example: bash submit_all.unique_cells.bash 2005
#

# Check that the year has been provided
if [ $# -ne 1 ]; then
    echo "Usage: submit_all.mcs_rainfall.bash <year>"
    exit 1
fi

# extract the year from the command line
year=$1
month=$2

# echo the year
echo "Finding mcs rainfall in year: $year"

# Set up months
months=(01 02 03 04 05 06 07 08 09 10 11 12)

# set up the extractor script
EXTRACTOR="/project/cssp_brazil/mcs_tracking_HG/analysis/submit.mcs_rainfall.sh"


precip_base_dir="/scratch/hgilmour/obs/precip"
mask_CCPF_base_dir="/project/cssp_brazil/mcs_tracking_HG/OBS_TRACKS/segmentation_obs"

# Set up the output directory
OUTPUT_DIR="/project/cssp_brazil/mcs_tracking_HG/lotus_output/mcs_rainfall"
mkdir -p $OUTPUT_DIR

# Loop over the months
for month in ${months[@]}; do
    
    echo $year
    echo $month

    # Find the precip files for the given month
    precip_file="precip_${year}_${month}.nc"
    # construct the precip path
    precip_path=${precip_base_dir}/${precip_file}

    # Find the mask files for the given month
    mask_CCPF_file="segmentation_yearly_${year}_INTERP_CCPF.nc" ## keep this as the whole year - it is split into months within the mcs_rainfall.py script
    # construct the precip path
    mask_CCPF_path=${mask_CCPF_base_dir}/${mask_CCPF_file}

    # Set up the output files
    OUTPUT_FILE="$OUTPUT_DIR/mcs_rainfall.$year.$month.out"
    ERROR_FILE="$OUTPUT_DIR/mcs_rainfall.$year.$month.err"

    # submit the batch job
    sbatch --mem=10000 --ntasks=2 --time=70 --output=$OUTPUT_FILE --error=$ERROR_FILE $EXTRACTOR $precip_path $mask_CCPF_path $month

done