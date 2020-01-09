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
pip install -e ./python/tflite2xcore/
```

If you made changes to the conda environment, export it (while activated) using:
```
conda env export --no-build | grep -Ev "^name:|^prefix:|libgcc-ng|libgfortran-ng|libstdcxx-ng|ai-tools" > environment.yml
```
Use `pip-autoremove` to uninstall unwanted `pip` packages.
This will clean up dependecies.

To remove the environment, deactivate and run:
```
conda remove -p ai_tools_venv/ --all
```

Install pipenv environment (deprecated)
---------------------------------------
The repo includes a Pipfile in case you run into difficulties with conda.

There have been issues with installing newer linux packages (tensorflow in particular):
https://github.com/pypa/pipenv/issues/3921


VSCode users
------------
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

Flatbuffers
-----------
You can experiment with tflite model flatbuffer files using the JSON format.
The flatbuffer schema compiler (`flatc`) can be used to convert the binary format to 
JSON text.

If you are planning to use `flatc` to convert `.tflite` models to/from JSON, be aware that it may lose precision on floating point numbers (see https://github.com/google/flatbuffers/issues/5371).
In particular, the python implementation of the graph transformer relies heavily on `flatc`.

The repo includes binaries under 'flatbuffers_xmos/flatc_{linux,darwin}' that fix the above issue, so you can use the graph transformer tool without having to build from source.
The tools in this repo should use these as defaults for the `flatc` binary.
Note that if you want to use another version (e.g. from your venv or system level install) you will have to specify that.

If you want to build from source, start by cloning the fork https://github.com/lkindrat-xmos/flatbuffers.
Then you can build using:

    > mkdir build
    > cd build/
    > ccmake ../
    > cmake ../
    > make
    > sudo make install

To convert a `.tflite` model flatbuffer file to JSON:

    > flatc --json path/to/tensorflow/tensorflow/lite/schema/schema.fbs -- path/to/your/model.tflite

To convert a JSON text file to `.tflite` model flatbuffer file:

    > flatc --binary path/to/tensorflow/tensorflow/lite/schema/schema.fbs path/to/your/model.json

See https://google.github.io/flatbuffers/flatbuffers_guide_using_schema_compiler.html for more information on `flatc`.
