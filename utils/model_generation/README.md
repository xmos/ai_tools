# Integration Tests

*Before running the integration tests, you must first build the `interpreter` and `test_model` example applications.*

## Running tests with the host interpreter

To run all tests using NUM threads

    > pytest integration_test -n NUM

To clear the cache and run all tests

    > pytest integration_test --cache-clear -n NUM

## Running tests with the device interpreter


## Other testing options

For additional logging output, including case index for failures, specify:

    --log-cli-level=debug

Run 

    > pytest -v

for more info on custom options.  


