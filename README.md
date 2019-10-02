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
If installing on a machine with CUDA GPUs, follow the instructions at https://www.tensorflow.org/install/gpu. Then you can install the conda environment via:
```
conda env create -p ./ai_tools_venv -f environment_gpu.yml
```

Activate the environment by passing in the path:
```
conda activate ai_tools_venv/
```

If you made changes to the conda environment, export it (while activated) using:
```
conda env export --no-build | grep -Ev "^name:|^prefix:|libgcc-ng|libgfortran-ng|libstdcxx-ng" > environment.yml
```
Use `pip-autoremove` to uninstall unwanted `pip` packages. This will clean up dependecies.

To remove the environment, deactivate and run:
```
conda remove -p ai_tools_venv/ --all
```

Install pipenv environment (deprecated)
-------
The repo includes a Pipfile in case you run into difficulties with conda.

There have been issues with installing newer linux packages (tensorflow in particular):
https://github.com/pypa/pipenv/issues/3921
