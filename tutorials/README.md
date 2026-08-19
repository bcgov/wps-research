
# BC Wildfire Mapping Tutorial

This folder contains tutorial for getting started with building wildfire perimeters, including Jupyter notebooks for data analysis and visualization.

## Table of Contents

* Requirements
* Environment Setup
* Running the Notebook
* Using GitHub Secrets
* Troubleshooting

### Requirements

* Python `3.10.12`
* Jupyter Notebook
* Dependencies listed in requirements.txt

### Environment Setup

1. Clone the repository:
```
git clone https://github.com/{your-username}/wps-research.git
cd wps-research
```


2. Create a virtual environment with the correct Python version and Jupyter already installed (In our example, we'll call our environment `cyberse_wildfire` but feel free so substitute this with any name you desire):
```
conda create -n cyberse_wildfire python==3.10.12 jupyter
```


3. Set the environment variable and make it persistent across sessions by replacing `your-project-name` with the actual Earth Engine Project name.
```
export CYBERSE=your-project-name

# Create a new file named env_vars.sh in this directory for the environment's activation script
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

# Edit this file and add your export command
touch $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export CYBERSE=your-project-name' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

Create a deactivation script and then edit it:
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
touch $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo 'unset CYBERSE' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
```

3. Activate the virtual environment and install the correct version of Python:

```
conda activate cyberse_wildfire
```


4. Install the required packages:
```
pip install -r requirements.txt
```


### Running the Notebook

1. Ensure your virtual environment is activated.

2. Start Jupyter Notebook using the command: `jupyter notebook BC_Wildfire_Mapping.ipynb`

3. Open the `BC_Wildfire_Mapping.ipynb` notebook and run the cells.



### Using GitHub Secrets

If you're running this notebook in a GitHub Actions workflow, the `CYBERSE` secret will be automatically set as an environment variable. Ensure that you've set up the secret in your GitHub repository:

1. Go to your GitHub repository

2. Click on "Settings" > "Secrets and variables" > "Actions"

3. Click "New repository secret"

4. Name it `CYBERSE` and paste your project name as the value

The GitHub Actions workflow will use this secret when running the notebook.



### Troubleshooting

If you encounter any issues:

1. Ensure you're using Python 3.10.12

2. Verify that all dependencies are correctly installed:
`pip list`

3. Check that the `CYBERSE` environment variable is set correctly:

On Windows: `echo %CYBERSE%`
On macOS and Linux: `echo $CYBERSE`



For any other issues, please open an issue in the GitHub repository.