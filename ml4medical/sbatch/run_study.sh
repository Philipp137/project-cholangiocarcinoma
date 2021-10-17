#!/bin/bash
for lr in  1e-4 1e-5 1e-6
do
    mkdir lr_$lr
    cd  lr_$L
    for freeze in -8 -6 -4 -2  
        do
        mkdir freeze_$freeze
        cd freeze_$freeze
        #cp /lustre/fs23/group/nic/stornati/bash_script .
        cp ../../bash_script .
	      touch scaling
        echo " /lustre/fs23/group/nic/stornati/julia/julia /lustre/fs23/group/nic/stornati/big_fss/fss.jl  $L  40 $Lambda  >> scaling " >> bash_script
        sbatch bash_script
        cd ..
        sleep 1
    done
    cd ..
done



