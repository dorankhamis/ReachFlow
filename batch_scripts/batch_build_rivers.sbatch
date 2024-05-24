#!/bin/bash 
#SBATCH --partition=short-serial-4hr
#SBATCH --account=short4hr
#SBATCH --job-name=build_river_objs
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=04:00:00
#SBATCH --array=0-250
#SBATCH --mem=20000

source /home/users/doran/software/envs/pytorch/bin/activate
python ../build_all_rivers.py ${SLURM_ARRAY_TASK_ID}

