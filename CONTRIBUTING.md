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

All python code should be [`blackened`](https://black.readthedocs.io/en/stable/).
For convenience, the default workspace settings file under `.vscode/` enables format-on-save, and `black` is also provided in the conda environments.

#### C, xC and ASM coding style

Changes to C, xC or ASM should be consistent with the style of existing C, xC and ASM code.

#### C++ coding style

Changes to C++ code should conform to the `.clang-format` file in the sub(project) or, if no such file is provided, to the
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

Use `clang-tidy` to check your C/C++ changes. To install `clang-tidy` on ubuntu:16.04, do:

```shell
apt-get install -y clang-tidy
```

You can check a C/C++ file by doing:


```shell
clang-format <my_cc_file> --style=google > /tmp/my_cc_file.cc
diff <my_cc_file> /tmp/my_cc_file.cc
```

### Running Tests

Before running the tests, ensure that you installed the correct pip packages in your environment:
```shell
pip install -e "./third_party/lib_tflite_micro/tflm_interpreter[test]"
pip install -e "./tflite2xcore[examples,test]"
```

To run all tests on 4 processor cores (replace 4 with the number of cores you want to use), run:
```shell
make test NUM_PROCS=4
```

On Linux you can utilize all your cores with
```shell
make test NUM_PROCS=$(grep -c ^processor /proc/cpuinfo)
```

On macOS you can utilize all your cores with
```shell
make test NUM_PROCS=$(system_profiler SPHardwareDataType | awk '/Total Number of Cores/{print $5}{next;}')
```

The [`Makefile`](Makefile) has a list of targets, including specific test and build targets.
Run the following for more information:
```shell
make help
```

To run the integration tests on the xcore.ai device, ensure that an explorer board is connected, then run:
```shell
pytest test/integration_test/ --cache-clear --use-device
```
Note that using multiple `pytest-xdist` workers when running on the divice is currently not supported.

## Development Tips

### git Submodules

After merging upstream changes, submodule repositories may need to be updated.
To update all submodules, run the following command:
```shell
make submodule_update
```

### Conda Environment

We strongly recommend using Conda if contributing to this project.

Install conda on your system if you don't already have it:
https://docs.conda.io/projects/conda/en/latest/user-guide/install/

It is recommended to configure Conda with the following options:
```shell
conda config --set auto_activate_base false
conda config --set env_prompt '({name})'
```

### VSCode Users

We recommend using a workspace-specific `settings.json` file along the lines of the following:
```json
{
    "files.exclude": {
        "ai_tools_venv": true,
        "**/.DS_Store": true,
        "**/.git/**": true,
        "**/.mypy_cache/**": true,
        "**/.ipynb_checkpoints/**": true,
        "**/__pycache__/**": true,
        "**/.pytest_cache/**": true,
        "**/*.egg-info/**": true,
        "**/.venv/**": true,
        "**/.build/**": true,
        "**/.lock*": true,
    },
    "files.watcherExclude": {
        "ai_tools_venv": true,
        "**/.DS_Store": true,
        "**/.git/**": true,
        "**/.mypy_cache/**": true,
        "**/.ipynb_checkpoints/**": true,
        "**/__pycache__/**": true,
        "**/.pytest_cache/**": true,
        "**/*.egg-info/**": true,
        "**/.venv/**": true,
        "**/.build/**": true,
        "**/.lock*": true,
        "**/build/**": true,
        "**/bin/**": true,
        "**/third_party/**": true,
    },
    "search.exclude": {
        "**/third_party": true,
    },
    "python.languageServer": "Pylance",
    "python.linting.pylintEnabled": true,
    "python.linting.mypyArgs": [
        "--config-file=${workspaceFolder}/mypy.ini"
    ],
    "python.linting.mypyCategorySeverity.error": "Information",
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "C_Cpp.clang_format_style": "file",
    "C_Cpp.clang_format_fallbackStyle": "Google",
    "editor.formatOnSave": true,
    "files.insertFinalNewline": true,
    "files.trimFinalNewlines": true,
}
```
