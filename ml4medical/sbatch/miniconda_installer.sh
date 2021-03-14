# First make a directory conda will be placed in
cd ~
#mkdir miniconda
# download miniconda
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda/miniconda.sh
# install miniconda
bash ~/miniconda/miniconda.sh -b -u -p ~/lib
rm -rf ~/miniconda/miniconda.sh
~/miniconda/bin/conda init bash
~/miniconda/bin/conda init zsh
