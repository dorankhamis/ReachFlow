#!/bin/bash 
#SBATCH --partition=par-single
#SBATCH --job-name=precipnormfeatures
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=48:00:00
#SBATCH -n 10
#SBATCH --array=0-29

source /home/users/doran/software/envs/pytorch/bin/activate
#python ./create_precip_normed_event_features.py ${SLURM_ARRAY_TASK_ID} 30 10 50 100
python ../create_precip_normed_event_features.py ${SLURM_ARRAY_TASK_ID} 30 10 0 0
