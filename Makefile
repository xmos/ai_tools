NUM_PROCS := 4

.DEFAULT_GOAL := help

#**************************
# xcore_interpreter targets
#**************************

.PHONY: xcore_interpreters_build
xcore_interpreters_build:
	cd python/xmos_ai_tools/xinterpreters/host/ && make install

#**************************
# xinterpreters smoke_test
#**************************

.PHONY: xinterpreters_smoke_test_host
xinterpreters_smoke_test_host:
	cd python/xmos_ai_tools/xinterpreters/host/ && make test

.PHONY: xinterpreters_smoke_test_device
xinterpreters_smoke_test_device:
	cd python/xmos_ai_tools/xinterpreters/device/ && make test

#**************************
# integration test targets
#**************************

.PHONY: xformer2_test
xformer2_integration_test:
	pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns/test_add -n 8 --dist loadfile --junitxml=integration_non_bnns_junit.xml
	pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns/test_avgpool2d -n 8 --dist loadfile --junitxml=integration_non_bnns_junit.xml
	pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns/test_conv2d -n 8 --dist loadfile --junitxml=integration_non_bnns_junit.xml
	pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns/test_conv2d_1x1 -n 8 --dist loadfile --junitxml=integration_non_bnns_junit.xml
	pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns/test_conv2d_shallowin -n 8 --dist loadfile --junitxml=integration_non_bnns_junit.xml
	pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns/test_custom_relu_conv2d -n 8 --dist loadfile --junitxml=integration_non_bnns_junit.xml
	pytest integration_tests/runner.py --models_path integration_tests/models/bnns/test_bconv2d_bin --bnn -n 8 --dist loadfile --junitxml=integration_bnns_junit.xml

#**************************
# default build and test targets
#**************************

.PHONY: build
build: xcore_interpreters_build

.PHONY: test
test: xinterpreters_smoke_test_host
test: xformer2_integration_test

#**************************
# other targets
#**************************

.PHONY: submodule_update
submodule_update: 
	git submodule update --init --recursive

.PHONY: clean
clean:
	cd python/xmos_ai_tools/xinterpreters/host/ && make clean

.PHONY: help
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

.PHONY: init_linux
init_linux:
	export BAZEL_VERSION=`cat experimental/xformer/.bazelversion` ;\
	curl -fLO "https://github.com/bazelbuild/bazel/releases/download/$${BAZEL_VERSION}/bazel-$${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
	chmod +x bazel-$${BAZEL_VERSION}-installer-linux-x86_64.sh && \
	./bazel-$${BAZEL_VERSION}-installer-linux-x86_64.sh --prefix=$$PWD/bazel

.PHONY: init_darwin
init_darwin:
	export BAZEL_VERSION=`cat experimental/xformer/.bazelversion` ;\
	curl -fLO "https://github.com/bazelbuild/bazel/releases/download/$${BAZEL_VERSION}/bazel-$${BAZEL_VERSION}-installer-darwin-x86_64.sh" && \
	chmod +x bazel-$${BAZEL_VERSION}-installer-darwin-x86_64.sh && \
	./bazel-$${BAZEL_VERSION}-installer-darwin-x86_64.sh --prefix=$$PWD/bazel

.PHONY: init_windows
init_windows:
	export BAZEL_VERSION=`cat experimental/xformer/.bazelversion` ;\
	curl -fLO 'https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-windows-x86_64.exe'
	mv bazel-${BAZEL_VERSION}-windows-x86_64.exe bazel.exe

.PHONY: build_release_linux
build_release_linux:
	(   module unload python && \
	    module load python/python-3.8.1 && \
	    module unload gcc && \
	    module load gcc/gcc-11.2.0 && \
	    module unload cmake && \
	    module load cmake/cmake-3.21.4 && \
	    python3 -m venv .venv && \
	    . .venv/bin/activate && \
	    pip install -r requirements.txt && \
	    cd experimental/xformer && ../../bazel/bin/bazel build --remote_cache=http://srv-bri-bld-cache:8080 //:xcore-opt --verbose_failures)
	rm -rf ../Installs/Linux/External/xformer
	mkdir -p ../Installs/Linux/External/xformer
	cp experimental/xformer/bazel-bin/xcore-opt ../Installs/Linux/External/xformer

.PHONY: build_release_darwin
build_release_darwin:
	( python3 -m venv .venv && \
	  . .venv/bin/activate && \
	  pip3 install --upgrade pip && \
	  pip3 install -r requirements.txt && \
	  cd experimental/xformer && ../../bazel/bin/bazel build --remote_cache=http://srv-bri-bld-cache:8080 --config=darwin_config //:xcore-opt --verbose_failures)
	rm -rf ../Installs/Mac/External/xformer
	mkdir -p ../Installs/Mac/External/xformer
	cp experimental/xformer/bazel-bin/xcore-opt ../Installs/Mac/External/xformer

.PHONY: build_release_windows
build_release_windows:
	python3 -m venv .venv
	(. .venv/bin/activate && pip install -r requirements.txt)
	(. .venv/bin/activate && cd experimental/xformer && ../../bazel build --remote_cache=http://srv-bri-bld-cache:8080 --config=windows_config //:xcore-opt --verbose_failures)
	mkdir -p ../Installs/Linux/External/xformer
	cp experimental/xformer/bazel-bin/xcore-opt ../Installs/Windows/External/xformer

TEST_SCRIPT= \
(cd xmos_ai_tools/src/xinterpreters/host/ && make build)&& \
(cd xmos_ai_tools && python3 setup.py bdist_wheel &&\
pip install ./xmos_ai_tools/dist/*"&& \
(cd experimental/xformer && ../../bazel/bin/bazel test --remote_cache=http://srv-bri-bld-cache:8080 //Test:all --verbose_failures)&& \
(pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns -n $(NUM_PROCS) --dist loadfile --junitxml=integration_non_bnns_junit.xml)&& \
(pytest integration_tests/runner.py --models_path integration_tests/models/bnns --bnn -n $(NUM_PROCS) --dist loadfile --junitxml=integration_bnns_junit.xml)

.PHONY: test_linux
test_linux:
	(. .venv/bin/activate && \
	    module unload python && \
	    module load python/python-3.8.1 && \
	    module unload gcc && \
	    module load gcc/gcc-11.2.0 && \
	    module unload cmake && \
	    module load cmake/cmake-3.21.4 && \
            ${TEST_SCRIPT} )

.PHONY: test_darwin
test_darwin:
	(. .venv/bin/activate && \
            ${TEST_SCRIPT} )
