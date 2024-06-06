#!/bin/bash
#SBATCH --mem=200000
#SBATCH --ntasks=4
#SBATCH --time=250

#Extract args from command line
ccpf_tracks_file=$1
tb_file=$2

# Print the tracks file
echo "$ccpf_tracks_file"
echo "$tb_file"

# Run the unique_cells.py script
python make_CCPF_masks.py ${ccpf_tracks_file} ${tb_file}