#!/bin/bash
#SBATCH --job-name=CCC-GPU
#SBATCH --output=CCC-GPU.out
#SBATCH --error=CCC-GPU.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=krah@math.tu-berlin.de
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=8
#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=100mb
#SBATCH --partition=gpu
#SBATCH --gpus:tesla:4
#SBATCH --time=00:10:00

module purge
module load cuda/10.0.130  intel/2018  openmpi/4.0.0 vasp/5.4.4

srun python main.py
