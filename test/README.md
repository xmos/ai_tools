# Integration Tests

*Before running the integration tests, you must first install the `xcore_interpreter` python module. Running `make develop` should take care of this.*

## Running all tests with the host interpreter

To run all tests in the suite:
> pytest integration_test

When modifying the test suite itself (or the model generation framework in `tflite2xcore`) it is usually necessary to clear cached models and data:
> pytest integration_test --cache-clear

The model cache is currently not process-safe. Thus when running tests on multiple processes with `pytest-xdist`, it is usually necessary to specify `--dist loadfile`.
To run all tests using NUM worker processes:
> pytest integration_test -n NUM --dist loadfile

## Running tests with the device interpreter

*To do this, the XMOS tools need to be installed and on the path.*

To run on the device, specify:

    --use-device

Do not use the `-n` option when running on the device.

## Other testing options

For additional logging output, including case index for failures, specify a desired log level with:

    --log-cli-level={warning,info,debug}

To halt at the first failure or error:

    -x

Run 

    > pytest -h

for more info on built-in and custom options.  
