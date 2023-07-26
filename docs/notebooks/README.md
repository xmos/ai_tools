# Notebooks

## Package Installation

### Docker (Recommended)

After installing Docker, simply run the `./start.sh` script to build and start the docker container. The logs should output the URL that jupyter lab can be accessed on. Copy and paste this URL into your web browser.

### Other

If you don't have docker installed, you can still run the notebooks. 

However, you must make sure that you  install the packages listed in `./requirements.txt` file in whichever kernel you are using for your jupyter lab.

It is important that the package versions match the `requirements.txt` file, so it is advised to create a separate kernel for this project.