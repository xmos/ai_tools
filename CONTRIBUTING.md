# Contributing Guidelines & Tips

## Pull Request Checklist
Before sending your pull requests, make sure you followed this list.

* Read [contributing guidelines and standards](CONTRIBUTING.md).
* Run [unit and integration tests](#Running-Tests).

## How to Contribute

### Contribution Guidelines and Standards

Before sending your pull request, make sure your changes are consistent with these guidelines and are consistent with the coding style used in this ai_tools repository.

#### General Guidelines and Philosophy For Contribution

* Include unit tests when you contribute new features, as they help to a) prove that your code works correctly, and b) guard against future breaking changes to lower the maintenance cost.
* Bug fixes also generally require unit tests, because the presence of bugs usually indicates insufficient test coverage.
* Keep API compatibility in mind when you change code.

#### Python coding style

Changes to Python code should conform to [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Use `pylint` to check your Python changes. To install `pylint` and check a file with `pylint`:

```bash
pip install pylint
pylint myfile.py
```

Pylint imposes different style guidelines than PEP8, so we recommend using the settings here: https://code.visualstudio.com/docs/python/linting#_default-pylint-rules

Before we have a detailed styleguide, consider also using `pycodestyle` to check for PEP8 compliance. If using `pycodestyle`, disable E501,E226,W503.

#### C, xC and ASM coding style

Changes to C, xC or ASM should be constant with the style of existing C, xC and ASM code.

#### C++ coding style

Changes to C++ code should conform to
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

Use `clang-tidy` to check your C/C++ changes. To install `clang-tidy` on ubuntu:16.04, do:

```bash
apt-get install -y clang-tidy
```

You can check a C/C++ file by doing:


```bash
clang-format <my_cc_file> --style=google > /tmp/my_cc_file.cc
diff <my_cc_file> /tmp/my_cc_file.cc
```

### Running Tests

There is a Makefile in the root directory that has some useful targets - especially when it comes to running tests before making a pull request.

* `all` - Run all the testing targets
* `lib_nn_test` - Build and run unit tests for lib_nn
* `tflite2xcore_test` - Run unit tests for tflite2xcore
* `integration_test` - Build any necessary components, generate all test data, and run all integration tests
* `clean` - Cleans all builds

To run the integration tests using multiple processors run 

  make integration_tests NUM_PROCS=4

Replace 4 with the number of processors you want to use.  

On Linux you can utilize all your cores with

  make integration_tests NUM_PROCS=$(grep -c ^processor /proc/cpuinfo)

On macOS you can utilize all your cores with

  make integration_test NUM_PROCS=$(system_profiler SPHardwareDataType | awk '/Total Number of Cores/{print $5}{next;}')

## Development Tips

### Installing Tools 15.0.0 Engineering Release

We currently require build 385 or newer.  Build 385 can be downloaded from http://srv-bri-jtools.xmos.local:8080/job/xmos-tools%20MANUAL/job/tools_installers/job/master/385/

NOTE: This link to the Engineering Release requires being inside the XMOS firewall.  We will replace this link once Tools 15 is officially released.

Linux users may want to manually add the SetEnv.sh script to the `.bashrc` file.  Add the following lines:

    XTIMEVER=Community_15.0.0
    pushd /usr/local/XMOS/xTIMEcomposer/$XTIMEVER/ > /dev/null
    source SetEnv
    popd > /dev/null


### git Submodules

At times submodule repositories will need to be updated.  To update all submodules, run the following command

> git submodule update --init --recursive

Users of older git versions, may need to use the following command

> git pull --recurse-submodules


### Conda Environment

If you made changes to the conda environment, export it (while activated) using:

> conda env export --no-build | grep -Ev "^name:|^prefix:|libgcc-ng|libgfortran-ng|libstdcxx-ng|ai-tools" > environment.yml

or for gpu, 

> conda env export --no-build | grep -Ev "^name:|^prefix:|libgcc-ng|libgfortran-ng|libstdcxx-ng|ai-tools" > environment_gpu.yml

Use `pip-autoremove` to uninstall unwanted `pip` packages. This will clean up dependecies.

To update your environment, run the following command

> conda env update --file environment.yml

or for gpu, 

> conda env update --file environment_gpu.yml

### VSCode Users

If you are using VS Code and conda, consider applying this fix:
https://github.com/microsoft/vscode-python/issues/3834#issuecomment-538016367

To suppress the annoying warning `"Unable to watch for file changes in this large workspace..."` add the following line to your `.vscode/settings.json`:

```
    "files.watcherExclude": {
      "**/.git/**": true,
      "**/.ipynb_checkpoints/**": true,
      "**/__pycache__/**": true,
      "**/.pytest_cache/**": true,
      "**/*.egg-info/**": true,
      "**/ai_tools_venv/**": true,
      "**/ai_tools_gpu_venv/**": true,
      "**/.venv/**": true,
      "**/.build/**": true,
      "**/.lock*": true,
      "**/build/**": true,
      "**/bin/**": true,
    },
```

To ensure that your linter is configured correctly add these lines to your `.vscode/settings.json`:
```
    "python.linting.pylintEnabled": true,
    "python.linting.pylintUseMinimalCheckers": true,
    "python.linting.pycodestyleEnabled": true,
    "python.linting.pycodestyleArgs": ["--ignore=E501,E226,W503"],
```
