.DEFAULT_GOAL := all

.PHONY: lib_nn_test
lib_nn_test:
	cd lib_nn/test/nn_operators && xmake
	cd lib_nn/test/nn_operators && xsim bin/nn_operators_test.xe

.PHONY: lib_nn_test_clean
lib_nn_test_clean:
	cd lib_nn/test/nn_operators && xmake clean

.PHONY: test_model
test_model: lib_nn
	cd examples/apps/test_model && make TARGET=x86
	cd examples/apps/test_model && make TARGET=xcore

.PHONY: test_model_clean
test_model_clean:
	cd examples/apps/test_model && make clean TARGET=x86
	cd examples/apps/test_model && make clean TARGET=xcore

.PHONY: tflite2xcore_test
tflite2xcore_test:
	cd tflite2xcore/tflite2xcore && pytest -v

.PHONY: integration_test
integration_test: test_model
	cd tests && ./generate_test_data.py
	cd tests && pytest -v --test-model=../examples/apps/test_model/bin/test_model
	cd tests && pytest -v

.PHONY: clean
clean: lib_nn_test_clean test_model_clean

.PHONY: test
test: lib_nn_test tflite2xcore_test integration_test

.PHONY: all
all: lib_nn test_model test