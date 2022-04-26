DISCLAIMER
--

This is the archived xformer1 integration tests. These are not currently used for testing, however, can be used for generating tflite models.


Generating models
--
Please build xformer using following the instructions mentioned [here](https://github.com/xmos/ai_tools#readme).
Then install tflite2xcore Python package using the following command as it is needed for generating models:
```shell
pip install "./archived/tflite2xcore[examples]"
```

To generate models for a particular test, use the following command defining the MODEL_DUMP_PATH env variable:
```shell
MODEL_DUMP_PATH=<path to a folder to dump models into> pytest archived/test/integration_test/test_directed/test_mobilenet_v1.py --cache-clear --only-experimental-xformer2
```

To dump all models, point pytest to the outermost dir:
```shell
MODEL_DUMP_PATH=<path to a folder to dump models into> pytest archived/test/integration_test --cache-clear --only-experimental-xformer2
```