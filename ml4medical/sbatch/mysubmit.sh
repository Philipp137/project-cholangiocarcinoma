#!/bin/bash
dir=$1
skriptname=$2
if [[ ! -e $dir ]]; then
    echo "creating $dir"
    mkdir $dir
elif [[ ! -d $dir ]]; then
    echo "$dir already exists but is not a directory" 1>&2
else
    echo "$dir already exists" 1>&2
    exit
fi

echo "copying data to $dir"
rsync -a --exclude={$dir,*.out,lightning_logs/,MSIMSS/,CCC/} . $dir
cd $dir 

echo "submit job: $skriptname"
sbatch $skriptname
