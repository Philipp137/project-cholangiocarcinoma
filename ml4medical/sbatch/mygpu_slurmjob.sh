#!/usr/local_rwth/bin/zsh
#SBATCH --job-name=CCC-GPU
###SBATCH --A
#SBATCH --output=CCC-GPU_id-%J.out
#SBATCH --error=CCC-GPU_id-%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=krah@math.tu-berlin.de
#SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4		        ## cpus 24 for pascal 48 for volta			
#SBATCH --ntasks-per-node=1   			## ntasks should be same as number of gpus	
#SBATCH --mem=32GB 				## 32 GB for gpus
###SBATCH --mem-per-cpu=100mb
#SBATCH --gres=gpu:volta:1
#SBATCH --time=4:00:00


source ~/.zshrc
conda activate ml4medical

for size in 398  454  518  591  674  769  877 1000
        do
        echo "runing batch size: $size " 
	python replaceInJson.py config_test.json trainer batch_size $size
	srun --mem=32GB --cpus-per-gpu=4 --gpus-per-node=volta:1 python main.py -c config_test.json
	##srun python main.py -c config.json
        sleep 1
done
wait
