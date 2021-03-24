#!/usr/local_rwth/bin/zsh
#SBATCH --job-name=CCC-GPU
###SBATCH --A
#SBATCH --output=CCC-GPU_id-%J.out
#SBATCH --error=CCC-GPU_id-%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=steffen.nitsch@live.de
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
###SBATCH --distribution=cyclic:cyclic
###SBATCH --mem-per-cpu=100mb
#SBATCH --gres=gpu:pascal:2
#SBATCH --time=00:10:00

source ~/.zshrc
conda activate ml4medical
srun python ../../main.py
