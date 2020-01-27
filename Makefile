.DEFAULT_GOAL := all

.PHONY: lib_nn
lib_nn:
	# FIXME: lib_nn build does not work yet
	#cd lib_nn/lib_nn && xwaf configure clean build

.PHONY: lib_nn_clean
lib_nn_clean:
	# TODO: Implement me!

.PHONY: lib_nn_test
lib_nn_test: lib_nn
	# TODO: Implement me!

.PHONY: test_model
test_model: lib_nn
	#cd examples/apps/test_model && make TARGET=x86
	cd examples/apps/test_model && make TARGET=xcore

.PHONY: test_model_clean
test_model_clean:
	#cd examples/apps/test_model && make clean TARGET=x86
	cd examples/apps/test_model && make clean TARGET=xcore

.PHONY: tflite2xcore_test
tflite2xcore_test:
	cd tflite2xcore/tflite2xcore && pytest

.PHONY: integration_test
integration_test: test_model
	cd tests && ./generate_test_data.py
	cd tests && pytest --test-model=../examples/apps/test_model/bin/test_model
	cd tests && pytest

.PHONY: clean
clean: lib_nn_clean test_model_clean

.PHONY: test
test: lib_nn_test tflite2xcore_test integration_test

.PHONY: all
all: lib_nn test_model test