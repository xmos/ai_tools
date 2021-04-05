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

Changes to C++ code should conform to
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

To run all tests on 4 processor cores (replace 4 with the number of cores you want to use), run:
```shell
make integration_tests NUM_PROCS=4
```

The Makefile in the root directory has a list of targets, including test and build targets.
Run the following for more information:
```shell
make help
```

On Linux you can utilize all your cores with
```shell
make integration_tests NUM_PROCS=$(grep -c ^processor /proc/cpuinfo)
```

On macOS you can utilize all your cores with
```shell
make integration_test NUM_PROCS=$(system_profiler SPHardwareDataType | awk '/Total Number of Cores/{print $5}{next;}')
```

## Development Tips

### git Submodules

At times submodule repositories will need to be updated.  To update all submodules, run the following command

> git submodule update --init

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

A default workspace settings file is included, but if desired it can be modified.
In this case, run `git update-index --assume-unchanged .vscode/settings.json` to prevent git from tracking local changes.
