# Integration Tests

*Before running the integration tests, you must first build the `xcore_interpreter` applications.*

## Running tests with the host interpreter

To run all tests using NUM worker processes

    > pytest integration_test -n NUM

To clear the cache and run all tests

    > pytest integration_test --cache-clear -n NUM

## Running tests with the device interpreter

To run on the device, specify:

    -- use-device

## Other testing options

For additional logging output, including case index for failures, specify:

    --log-cli-level=debug

To halt at the first failure or error:

    -x

Run 

    > pytest -h

for more info on custom options.  
