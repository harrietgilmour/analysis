#!/bin/bash
#SBATCH --mem=10000
#SBATCH --ntasks=2
#SBATCH --time=70

#Extract args from command line
precip_file=$1
mask_CCPF_file=$2
month=$3

# Print the tracks file
echo "$precip_file"
echo "$mask_CCPF_file"
echo "$month"

# Run the unique_cells.py script
python mcs_rainfall.py ${precip_file} ${mask_CCPF_file} ${month}
