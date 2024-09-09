#!/bin/bash
#SBATCH -A m1800_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"

# source /global/homes/p/peishi89/env_jax_canveg.sh
module load python/3.9-anaconda-2021.11
module load cudatoolkit
conda activate jax-watershed
export PYTHONPATH=${PYTHONPATH}:/global/cfs/cdirs/m1800/peishi/JAX-CanVeg/src

srun -n 1 -c 1 python ./US-Bi1/train_models.py &
srun -n 1 -c 1 python ./US-Hn1/train_models.py &
srun -n 1 -c 1 python ./US-Me2/train_models.py &
srun -n 1 -c 1 python ./US-Whs/train_models.py &
wait