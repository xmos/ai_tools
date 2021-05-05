NUM_PROCS := 4

.DEFAULT_GOAL := help

#**************************
# xcore_interpreter targets
#**************************

.PHONY: xcore_interpreters_build
xcore_interpreters_build:
	cd utils/adf/ && make build

#**************************
# tflite2xcore targets
#**************************

.PHONY: lib_flexbuffers_build
lib_flexbuffers_build:
	cd utils/lib_flexbuffers && bash build.sh

.PHONY: tflite2xcore_unit_test
tflite2xcore_unit_test:
	tflite2xcore/tflite2xcore/tests/runtests.py tflite2xcore/tflite2xcore/tests -n $(NUM_PROCS) --junit

.PHONY: tflite2xcore_dist
tflite2xcore_dist:
	cd tflite2xcore && bash build_dist.sh

#**************************
# integration test targets
#**************************

.PHONY: integration_test
integration_test:
	cd test && pytest integration_test --cache-clear --collect-only -qq
	cd test && pytest integration_test -n $(NUM_PROCS) --dist loadfile --junitxml=integration_junit.xml

.PHONY: xformer2_test
xformer2_test:
	cd test && pytest integration_test --cache-clear --collect-only -qq
	cd test && pytest integration_test -n $(NUM_PROCS) --dist loadfile --experimental-xformer2 --junitxml=integration_junit.xml

#**************************
# default build and test targets
#**************************

.PHONY: build
build: lib_flexbuffers_build xcore_interpreters_build

.PHONY: test
test: tflite2xcore_unit_test integration_test

#**************************
# other targets
#**************************

.PHONY: submodule_update
submodule_update: 
	git submodule update --init --recursive

.PHONY: clean
clean:
	cd utils/adf/ && make clean
	rm -rf utils/lib_flexbuffers/build

.PHONY: help
help:
	@:  # This silences the "Nothing to be done for 'help'" output
	$(info usage: make [target])
	$(info )
	$(info )
	$(info primary targets:)
	$(info   build                         Build all components)
	$(info   test                          Run all tests (requires tflite2xcore[test] package))
	$(info   clean                         Clean all build artifacts)
	$(info )
	$(info secondary targets:)
	$(info   lib_flexbuffers_build         Build lib_flexbuffers)
	$(info   xcore_interpreters_build      Build xcore_interpreters)
	$(info   tflite2xcore_unit_test        Run tflite2xcore unit tests (requires tflite2xcore[test] package))
	$(info   tflite2xcore_dist             Build tflite2xcore distribution (requires tflite2xcore[test] package))
	$(info   integration_test              Run integration tests (requires tflite2xcore[test] package))
	$(info   xformer2_test                 Run integration tests with xformer2 (experimental requires tflite2xcore[test] package))
	$(info )
