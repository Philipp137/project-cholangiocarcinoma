#!/usr/local_rwth/bin/zsh

## slurm submit script
#SBATCH --job-name=CCC_mainz
#SBATCH --account=rwth0777
#SBATCH --output=CCC-%J.out
###SBATCH --output=test.out
###SBATCH --error=CCC-%J.out
#SBATCH --mail-type=END
##SBATCH --mail-user=krah@math.tu-berlin.de, steffen.buechholz@outlook.de
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12       
#SBATCH --ntasks=1   		
#SBATCH --mem=0 				## 32 GB for gpus
#SBATCH --gres=gpu:volta:2                      ## request 2 gpus per node
#SBATCH --time=2:00:00


source ~/.zshrc
# -------------------------
# debugging flags (optional)
# export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0,1,2,3

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest cuda
#module load nvhpc/21.2-cuda10.2   
#module load cuda/110
#module load cudnn/8.0.5
# ------------------------
conda activate ml4medical

srun python main.py -c config_CCC.json
#srun python show_val_heatmap_CCC.py
