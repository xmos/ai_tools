NUM_PROCS := 8
.DEFAULT_GOAL := help
.PHONY: xcore_interpreters_build xinterpreters_smoke_test_host xformer2_test version_check build test submodule_update clean help

xcore_interpreters_build:
	$(MAKE) -C python/xmos_ai_tools/xinterpreters/host/ install

xformer2_integration_test:
	pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns -n $(NUM_PROCS) --junitxml=integration_non_bnns_junit.xml
	pytest integration_tests/runner.py --models_path integration_tests/models/bnns --bnn -n $(NUM_PROCS) --junitxml=integration_bnns_junit.xml

version_check:
	cd ./xformer && ./version_check.sh

build: version_check xcore_interpreters_build

test: xformer2_integration_test

submodule_update: 
	git submodule update --init --recursive

clean:
	$(MAKE) -C python/xmos_ai_tools/xinterpreters/host/ clean

help:
	@:  # This silences the "Nothing to be done for 'help'" output
	$(info usage: make [target])
	$(info )
	$(info )
	$(info primary targets:)
	$(info   build                         Build all components)
	$(info   test                          Run all tests)
	$(info   clean                         Clean all build artifacts)
	$(info )
	$(info secondary targets:)
	$(info   xcore_interpreters_build      Build xcore_interpreters)
	$(info   xformer2_integration_test     Run integration tests with xformer2)
	$(info )
