#!/usr/local_rwth/bin/zsh
#SBATCH --job-name=CCC-GPU
###SBATCH --A
#SBATCH --output=CCC-GPU_id-%J.out
#SBATCH --error=CCC-GPU_id-%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=krah@math.tu-berlin.de
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
###SBATCH --distribution=cyclic:cyclic
###SBATCH --mem-per-cpu=100mb
#SBATCH --gres=gpu:volta:1
#SBATCH --time=00:00:5

source .zshrc
conda --version
###srun python main.py
