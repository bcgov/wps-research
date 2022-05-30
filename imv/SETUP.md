# Setup:
1. Install Oracle virtualbox https://www.virtualbox.org/wiki/Downloads under VirtualBox x.x.xx platform packages
2. Create a VM inside VirtualBox (type: Linux, Ubuntu 64bit) by downloading https://releases.ubuntu.com/20.04.2.0/ubuntu-20.04.2.0-desktop-amd64.iso
and using it as your "startup disk" while using VirtualBox to create a new Virtual Machine
* call your VM something creative like VM; use default options and write down your username / password 
* use the default memory settings; try to adjust them later if they're not working
* use VDI format virtual hard disk / dynamically allocated e.g. 10GB initial size
* minimal installation for Ubuntu is fine!

3. Boot your VM and install "guest additions" for your host operating system e.g. Windows host, ubuntu guest: https://www.tecmint.com/install-virtualbox-guest-additions-in-ubuntu/. Then reboot your VM
4. Make sure that your VM window scales up to fit your screen. This may involve some fiddling in display settings (on your VM desktop or the settings for the VM in Virtualbox menu). You also likely want to enable the clipboard (at least into the VM), and perhaps a shared folder to move in data
5. Under "activities" in top bar, search for Terminal (run it and add to favourites as well)
6. Within Terminal, you should be in your home folder already that is:

```bash
cd ~
```
should do nothing

7. Install some "standard" packages:
```bash
sudo apt install git wget gdal-bin gcc g++ python3 python3-matplotlib python3-sklearn python3-scipy freeglut3-dev 
```
8. Make sure packages are up to date
```bash
sudo apt update && sudo apt upgrade
```
9. Create a folder called GitHub:
```bash
mkdir GitHub
```
10. Enter the folder:
```bash
cd GitHub
```
11. "Clone" the repo (download the source code and test data): method for if you have ssh keys set up already:
```bash
git clone git@github.com:bcgov/wps-research.git
```

method for not having ssh keys set up:

```bash
wget https://github.com/bcgov/wps-research/archive/master.zip
unzip master
mv wps-research-master bcws-psu-research
```

12. Enter the repo:
```bash
cd wps-research
```
13. Navigate to the MVP software interface:
```bash
cd imv
```
14. Compile MVP viewer...
```
python3 compile.py # compile the viewer to /usr/bin/imv
```

15. navigate to test data and run
'''
cd peppers
imv
'''

* **Important: always click somewhere on full-scene/ overview window, first to buffer data under it**
