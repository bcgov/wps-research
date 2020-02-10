# install sen2cor and sen2mosaic, taken from: https://sen2mosaic.readthedocs.io/en/latest/setup.html

# install anaconda 
wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
chmod +x Anaconda3-2019.03-Linux-x86_64.sh
./Anaconda3-2019.03-Linux-x86_64.sh

# set up anaconda
conda create -n sen2mosaic -c conda-forge python=3.7 scipy pandas psutil scikit-image gdal opencv pyshp
conda activate sen2mosaic

# install sen2cor
wget http://step.esa.int/thirdparties/sen2cor/2.8.0/Sen2Cor-02.08.00-Linux64.run
chmod +x Sen2Cor-02.08.00-Linux64.run
./Sen2Cor-02.08.00-Linux64.run

# reference bashrc
echo "source ~/Sen2Cor-02.08.00-Linux64/L2A_Bashrc" >> ~/.bashrc
exec -l $SHELL

# install sentinelsat
python3 -m pip install sentinelsat

# install sen2mosaic
git clone https://sambowers@bitbucket.org/sambowers/sen2mosaic.git
cd sen2mosaic
python3 setup.py install

# reference bashrc
echo "alias s2m='_s2m() { python ~/sen2mosaic/cli/\"\$1\".py \$(shift; echo \"\$@\") ;}; _s2m'" >> ~/.bashrc




