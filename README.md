AI/ML tools repository
======================

Installation
------------
`xmos-ai-tools` is available on [PyPi](https://pypi.org/project/xmos-ai-tools/).
It includes:
- the MLIR-based XCore optimizer(xformer) to optimize Tensorflow Lite models for XCore
- the XCore tflm interpreter to run the transformed models on host
- the XCore tflm interpreter to run the transformed models on an xcore device connected over usb or spi

It can be installed with the following command:
```shell
pip install xmos-ai-tools
```
If you want to install the latest development version, use:
```shell
pip install xmos-ai-tools --pre
```

Installing `xmos-ai-tools` will make the `xcore-opt` binary available in your shell to use directly, or you can use the Python interface as detailed [here](https://pypi.org/project/xmos-ai-tools/).

Building xmos-ai-tools
----------------------
Some dependent libraries are included as git submodules.
These can be obtained by cloning this repository with the following commands:
```shell
git clone git@github.com:xmos/ai_tools.git
cd ai_tools
make submodule_update
```

Install at least version 15 of the XMOS tools from your preferred location and activate it by sourcing `SetEnv` in the installation root.

[CMake 3.14](https://cmake.org/download/) or newer is required for building libraries and test firmware.
A correct version of CMake (and `make`) is included in the Conda environment file, utils/environment.yml.
To set up and activate the environment, simply run:
```shell
conda env create -p ./ai_tools_venv -f environment.yml
conda activate ai_tools_venv/
```

Build the XCore host tflm interpreter libraries with default settings (see the [`Makefile`](Makefile) for more), run:
```shell
make build
```

Install the necessary python packages and the `xtftlm_interpreter` python package using `pip` (inside the conda venv):
```shell
pip install -r "./requirements.txt"
```

After following the above instructions, to build xformer, please follow the build instructions [here](https://github.com/xmos/ai_tools/tree/develop/experimental/xformer#readme)
