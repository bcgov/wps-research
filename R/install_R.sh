# Install R v. 4 on ubuntu
# from:
#   https://rtask.thinkr.fr/installation-of-r-4-0-on-ubuntu-20-04-lts-and-tips-for-spatial-packages/
sudo apt install update-manager-core
sudo apt upgrade
sudo apt dist-upgrade
sudo apt autoremove
sudo do-release-upgrade
sudo apt-get purge r-base* r-recommended r-cran-*
sudo apt autoremove
sudo apt update
sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo apt update
gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
gpg -a --export E298A3A825C0D65DFD57CBB651716619E084DAB9 | sudo apt-key add -
sudo apt install r-base r-base-core r-recommended r-base-dev

# may need to update R packages with: run "sudo R" then enter:
#  update.packages(ask = FALSE, checkBuilt = TRUE)
