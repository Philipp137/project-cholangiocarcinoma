#!/usr/local_rwth/bin/zsh

## slurm submit script
#SBATCH --job-name=badtiles_val
#SBATCH --account=rwth0777
#SBATCH --output=badtiles.out
##SBATCH --error=MSIMSS_id-%J.out
#SBATCH --mail-type=END
##SBATCH --mail-user=krah@math.tu-berlin.de, steffen.buechholz@outlook.de
#SBATCH --nodes=1
#SBATCH --ntasks=1   		                # this will spawn the same job ntasks times
#SBATCH --cpus-per-task=24   		
#SBATCH --mem=0 				## 32 GB for gpus
##SBATCH --gres=gpu:volta:2                      ## request 2 gpus per node
#SBATCH --time=01:00:00


source ~/.zshrc


# might need the latest cuda
#module load nvhpc/21.2-cuda10.2   
#module load cuda/110
#module load cudnn/8.0.5
# ------------------------
conda activate ml4medical

python ml4medical/normalize.py
#srun python show_val_heatmap_CCC.py
