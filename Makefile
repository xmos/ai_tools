NUM_PROCS := 4
CLOBBER_FLAG := '-c'

.DEFAULT_GOAL := help

#**************************
# xcore_interpreter targets
#**************************

.PHONY: xcore_interpreters_build
xcore_interpreters_build:
	cd utils/adf/ && make build CLOBBER_FLAG=$(CLOBBER_FLAG)

#**************************
# tflite2xcore targets
#**************************

.PHONY: lib_flexbuffers_build
lib_flexbuffers_build:
	cd utils/lib_flexbuffers && bash build.sh $(CLOBBER_FLAG)

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

#**************************
# ALL build target
#**************************

.PHONY: build
build: lib_flexbuffers_build \
 xcore_interpreters_build

#**************************
# ALL tests target
#**************************

.PHONY: test
test: tflite2xcore_unit_test \
 integration_test

#**************************
# development targets
#**************************

.PHONY: submodule_update
submodule_update: 
	git submodule update --init --recursive

.PHONY: _develop
_develop: submodule_update build

.PHONY: develop
develop: CLOBBER_FLAG=''
develop: _develop

.PHONY: clobber
clobber: CLOBBER_FLAG='-c'
clobber: _develop

.PHONY: help
help:
	@:  # This silences the "Nothing to be done for 'help'" output
	$(info usage: make [target])
	$(info )
	$(info )
	$(info primary targets:)
	$(info   build                         Build all components)
	$(info   develop                       Update submodules and build all components)
	$(info   clobber                       Update submodules, then clean and rebuild all components)
	$(info   test                          Run all tests (requires tflite2xcore[test] package))
	$(info )
	$(info secondary targets:)
	$(info   lib_flexbuffers_build         Build lib_flexbuffers)
	$(info   xcore_interpreters_build      Build xcore_interpreters)
	$(info   tflite2xcore_unit_test        Run tflite2xcore unit tests (requires tflite2xcore[test] package))
	$(info   tflite2xcore_dist             Build tflite2xcore distribution (requires tflite2xcore[test] package))
	$(info   integration_test              Run integration tests (requires tflite2xcore[test] package))
	$(info )
