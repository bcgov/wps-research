USER_HOME=$(getent passwd "${SUDO_USER:-$USER}" | cut -d: -f6)

# make a subdirectory inside it
mkdir -p "$USER_HOME/sen2cor"
cd "$USER_HOME/sen2cor"

wget http://step.esa.int/thirdparties/sen2cor/2.5.5/Sen2Cor-02.05.05-Linux64.run

# wget https://step.esa.int/thirdparties/sen2cor/2.12.0/Sen2Cor-02.12.03-Linux64.run

# chmod 755 Sen2Cor-02.12.03-Linux64.run
chmod 755 Sen2Cor-02.05.05-Linux64.run
# ./Sen2Cor-02.12.03-Linux64.run 
./Sen2Cor-02.05.05-Linux64.run
