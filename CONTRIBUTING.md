# Installing Tools 15.0.0 Engineering Release

We currently require build 310 or newer.  Build 310 can be downloaded from http://srv-bri-jtools:8080/job/xmos-tools%20MANUAL/job/tools_installers/job/master/310/

Linux users will need to manually add the SetEnv.sh script to the .bashrc file.  Add the following lines

    XTIMEVER=Community_15.0.0_eng
    pushd /usr/local/XMOS/xTIMEcomposer/$XTIMEVER/ > /dev/null
    source SetEnv
    popd > /dev/null


# git Submodules

At times submodule repositories will need to be updated.  To update all submodules, run the following command

> git submodule update --init --recursive


# Conda Environment

If you made changes to the conda environment, export it (while activated) using:

> conda env export --no-build | grep -Ev "^name:|^prefix:|libgcc-ng|libgfortran-ng|libstdcxx-ng|ai-tools" > environment.yml

or for gpu, 

> conda env export --no-build | grep -Ev "^name:|^prefix:|libgcc-ng|libgfortran-ng|libstdcxx-ng|ai-tools" > environment_gpu.yml

Use `pip-autoremove` to uninstall unwanted `pip` packages. This will clean up dependecies.

To update your environment, run the following command

> conda env update --file environment.yml

or for gpu, 

> conda env update --file environment_gpu.yml

# VSCode Users

If you are using VS Code and conda, consider applying this fix:
https://github.com/microsoft/vscode-python/issues/3834#issuecomment-538016367

To suppress the annoying warning `"Unable to watch for file changes in this large workspace..."` add the following line to your `.vscode/settings.json`:
```
    "files.watcherExclude": {
      "**/.git/**": true,
      "**/.ipynb_checkpoints/**": true,
      "**/__pycache__/**": true,
      "**/ai_tools_venv/**": true,
      "**/ai_tools_gpu_venv/**": true,
      "**/.venv/**": true,
      "**/.build/**": true,
      "**/.lock*": true,
      "**/build/**": true,
      "**/bin/**": true,
    },
```