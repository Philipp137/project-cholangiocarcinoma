#!/usr/local_rwth/bin/zsh
#SBATCH --job-name=CCC-GPU
###SBATCH --A
#SBATCH --output=CCC-GPU_id-%J.out
#SBATCH --error=CCC-GPU_id-%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=steffen.nitsch@live.de
#SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=4		        ## cpus 24 for pascal 48 for volta			
#SBATCH --ntasks-per-node=1   			## ntasks should be same as number of gpus	
#SBATCH --mem=0 				## 32 GB for gpus
###SBATCH --mem-per-cpu=100mb
#SBATCH --gres=gpu:volta:1
#SBATCH --time=02:00:00


source ~/.zshrc
conda activate ml4medical
srun python ../../main.py
