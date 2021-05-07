# Integration Tests

Before running the integration tests, you must first install the `xcore_interpreter` python module as described in the main [`README.md`](../README.md).

## Running all tests with the host interpreter

To run all tests in the suite:
```
pytest integration_test
```

When modifying the test suite itself (or the model generation framework in `tflite2xcore`) it is usually necessary to clear cached models and data:
```
pytest integration_test --cache-clear
```

The model cache is currently not process-safe. Thus when running tests on multiple processes with `pytest-xdist`, it is usually necessary to specify `--dist loadfile`.
To run all tests using NUM worker processes:
```
pytest integration_test -n NUM --dist loadfile
```

## Running tests with the device interpreter

*To do this, the XMOS tools need to be installed and on the path.*

To run on the device, specify the `--use-device` flag for any test. Some tests are skipped on the device to speed up execution or avoid known errors associated with insufficient arena size or other HW-related problems.

Do not use the `-n` option when running on the device.

## Running specific tests or configurations

To run a specific test (function) in a module, use the `::` notation, e.g.:
```
pytest integration_test/test_single_op_models/test_fully_connected.py::test_output
```

Test modules typically include multiple configurations, either defined by the `CONFIGS` variable in the module, or in an adjacent `.yml` file with identical base name (e.g. `test_fully_connected.{py,yml}`).
To run a single configuration with index `0` of a particular test (function):
```
pytest integration_test/test_single_op_models/test_fully_connected.py::test_output[CONFIGS[0]]
```

## Dump test models

Specifying `--dump models` will save the converted `.tflite` models in the model cache directory. To print the location of these saved models, set the logging level to `info` by setting `--log-cli-level=info`.

## Other testing options

For additional logging output, including case index for failures, specify a desired log level with `--log-cli-level={warning,info,debug}`.

To halt at the first failure or error use the `-x` flag.

For more info on built-in and custom options, run `pytest -h`, or consult the documentation https://docs.pytest.org/en/stable/contents.html.  
