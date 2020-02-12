AI/ML tools repository
======================

Summary
-------


Install conda environment
-------------------------
Install conda on your system if you don't already have it:
https://docs.conda.io/projects/conda/en/latest/user-guide/install/

It is recommended to configure conda with the following options:
```
conda config --set auto_activate_base false
conda config --set env_prompt '({name})'
```

It is recommended that you install the virtual environment in the repo's directory:
```
cd ai_tools
conda env create -p ./ai_tools_venv -f environment.yml
```
If installing on a machine with CUDA GPUs, follow the instructions at https://www.tensorflow.org/install/gpu.
Then you can install the conda environment via:
```
conda env create -p ./ai_tools_gpu_venv -f environment_gpu.yml
```

Activate the environment by specifying the path, then install the ai-tools python package:
```
conda activate ai_tools_venv/
pip install -e tflite2xcore/
```

To remove the environment, deactivate and run:
```
conda remove -p ai_tools_venv/ --all
```

Submodules
----------
Some dependent libraries are included as git submodules. These can be obtained by cloning this repository with the following command:

> git clone --recurse-submodules git@github.com:xmos/ai_tools.git
