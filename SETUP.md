# Setup:
1. Install Oracle virtualbox https://www.virtualbox.org/wiki/Downloads under VirtualBox x.x.xx platform packages
2. Create a VM inside VirtualBox (type: Linux, Ubuntu 64bit) by downloading https://releases.ubuntu.com/20.04.2.0/ubuntu-20.04.2.0-desktop-amd64.iso
and using it as your "startup disk" while using VirtualBox to create a new Virtual Machine
* call your VM something creative like VM; use default options and write down your username / password 
* use the default memory settings; try to adjust them later if they're not working
* use VDI format virtual hard disk / dynamically allocated e.g. 10GB initial size
* minimal installation for Ubuntu is fine!

3. Boot your VM and install "guest additions" for your host operating system e.g. Windows host, ubuntu guest: https://www.tecmint.com/install-virtualbox-guest-additions-in-ubuntu/
4. Under "activities" in top bar, search for Terminal (run it and add to favourites)
5. Within Terminal, Navigate to your home folder:
```bash
cd ~
```
5. Create a folder called GitHub:
```bash
mkdir GitHub
```
6. Enter the folder and clone the repo:
```bash
cd GitHub
```

7. Clone the repo:
```bash
git clone git@github.com:bcgov/bcws-psu-research.git
```

8. Enter the repo:
```bash
cd bcws-psu-research
```

9. Navigate to the MVP software interface:
```bash
cd imv
```
10. Install some "standard" packages:
```bash
sudo apt install python3 python3-matplotlib python3-sklearn python3-scipy gcc g++ freeglut3-dev
```

11. Before we run it, a little housekeeping: upgrade packages
```bash
sudo apt update && sudo apt upgrade
```

12. Compile MVP viewer...
```
sudo touch /usr/bin/imv # this is where the command will go!

sudo chmod 755 /usr/bin/imv # make it runnable!

python3 compile.py # compile the viewer and overwrite /usr/bin/imv
```

12. navigate to peppers test data and run:
'''
cd peppers
imv
'''

* Session: always click somewhere on full-scene/ overview window, first to buffer data under it
