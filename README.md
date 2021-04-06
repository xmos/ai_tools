AI/ML tools repository
======================

Summary
-------


Installation
------------
Some dependent libraries are included as git submodules.
These can be obtained by cloning this repository with the following command:
```shell
git clone git@github.com:xmos/ai_tools.git
cd ai_tools
git submodule update --init
```

Install at least version 15 of the XMOS tools from your preferred location and activate it by sourcing `SetEnv` in the installation root.

[CMake 3.14](https://cmake.org/download/) or newer is required for building libraries and test firmware.
A correct version of CMake (and `make`) is included in the [`adf`](https://github.com/xmos/adf) submodule's Conda environment file, [`environment.yml`](https://github.com/xmos/adf/blob/develop/environment.yml).
To set up and activate the environment, simply run:
```shell
conda env create -p ./ai_tools_venv -f utils/adf/environment.yml
conda activate ai_tools_venv/
```

If installing on a machine with CUDA GPUs, follow the instructions at https://www.tensorflow.org/install/gpu#software_requirements to install the necessary drivers and libraries.
```shell
TODO: add and test instructions for GPU builds.
```

Build the libraries with default settings (see the [`Makefile`](Makefile) for more), run:
```shell
make build
```

Install the `xcore_interpreters` and `tflite2xcore` python packages using `pip` (preferably inside a venv):
```shell
pip install -e "./utils/adf/xcore_interpreters"
pip install -e "./tflite2xcore[examples]"
```

Docker Image
------------

The Dockerfile provided is used in the CI system but can serve as a guide to system setup.
Installation of the XMOS tools requires connection to our network.

```shell
docker build -t xmos/ai_tools .
docker run -it \
    -v $(pwd):/ws \
    -u $(id -u):$(id -g)  \
    -w /ws  \
    xmos/ai_tools \
    bash -l
```

Note that this container will stop when you exit the shell
For a persistent container:
 - add `-d` to the docker run command to start detached;
 - add `--name somename`;
 - enter with `docker exec -it somename bash -l`;
 - stop with `docker stop somename`.

Then inside the container
```shell
# setup environment
conda env create -p ai_tools_venv -f utils/adf/environment.yml
/XMOS/get_tools.py 15.0.1
conda activate ./ai_tools_venv
pip install -e "./xcore_interpreters[test]"
pip install -e "./tflite2xcore[examples,test,dev]"
# activate tools (each new shell)
module load tools/15.0.1
# build all and run tests
make ci
```
