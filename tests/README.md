# Integration Tests

Before running the integration tests, you must first build the `test_model` example application
and generate the test data.

To build the `test_model` applications, follow the instructions located in:

    utils/test_model

To generate the integration test data run

    > ./generate_test_data.py

Note, generate_test_data.py takes a few minutes to complete.

By default, all tests are configured to run using the xCORE simulator.  You must have the 
xCORE toolchain installed and in your system path.  

To run the tests on the host PC

    > pytest --test-app=../utils/test_model/bin/test_model_cmdline -n NUM

To run the tests on the xCORE simulator

    > pytest --test-app=../utils/test_model/bin/test_model_cmdline.xe -n NUM

Useful, custom command line options.

    --test-dir path/to/non-default/test/data
    --abs-tol 2
    --max-count 1

To run a single test

    > pytest -k name_of_test


 Run 

    > pytest -v

for more info on custom options.  


