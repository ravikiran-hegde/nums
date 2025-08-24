#!/bin/bash
#SBATCH --job-name=sim_earth
#SBATCH --time=08:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=mh0081
#SBATCH --output=%x_%j_output.log
#SBATCH --error=%x_%j_error.log

module load python3
source activate arts3-env # change the name for the environment
python -u sim_earth.py $1
