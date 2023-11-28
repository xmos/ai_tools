// Things to optimise if this is too slow:
// - build device runtime in parallel with host runtimes, use mutex before combining into wheel
// - install bazelisk directly on the Jenkins machines: will save about 50MB of downloads per CI run

@Library('xmos_jenkins_shared_library@v0.25.0') _

getApproval()

pipeline {
    environment {
        BAZEL_CACHE_URL = 'http://srv-bri-bld-cache:8080'
        BAZEL_USER_ROOT = "${WORKSPACE}/.bazel/"
        REBOOT_XTAG = '1'
    }
    parameters {
        string(
            name: 'TOOLS_VERSION',
            defaultValue: '15.2.1',
            description: 'The tools version to build with (check /projects/tools/ReleasesTools/)'
        )
        string( 
            name: 'TAG_VERSION',
            defaultValue: '',
            description: 'The release version, leave empty to not publish a release'
        )
    }

    options {
        timestamps()
        skipDefaultCheckout()
        buildDiscarder(logRotator(
            numToKeepStr:         env.BRANCH_NAME ==~ /develop/ ? '100' : '',
            artifactNumToKeepStr: env.BRANCH_NAME ==~ /develop/ ? '100' : ''
        ))
    }
    agent { label "linux && 64 && !noAVX2" } 
    stages {
        stage("Build device runtime") { steps {
                setupRepo()
                createVenv("requirements.txt")
                withVenv { sh "pip install -r requirements.txt" }
                withVenv { withTools(params.TOOLS_VERSION) { createZip("device") } }
                dir("third_party/lib_tflite_micro/build/") {
                    stash name: "release_archive", includes: "release_archive.zip"
                }
        } }
        stage("Build host wheels") {
            parallel {
                stage("Build linux runtime") { steps {
                    withTools(params.TOOLS_VERSION) { createZip("linux") }
                    extractRuntime()
                    buildXinterpreter() 
                    script {
                        docker.image('tensorflow/build:2.14-python3.9').inside("-e SETUPTOOLS_SCM_PRETEND_VERSION=${env.TAG_VERSION} -v ${env.WORKSPACE}:/ai_tools -w /ai_tools") {
                            dir("xformer") {
                                // sh "wget https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64"
                                sh "mkdir -p ai_tools/.cache"
                                // sh "chmod +x bazelisk-linux-amd64"
                                sh "bazel build //:xcore-opt --verbose_failures --linkopt=-lrt --crosstool_top='@sigbuild-r2.14-clang_config_cuda//crosstool:toolchain' --//:disable_version_check --output_user_root=ai_tools/.bazel build --remote_cache=${env.BAZEL_CACHE_URL} --disk_cache=ai_tools/.cache"
                            }
                            dir("python") {
                                sh "pip install auditwheel==5.2.0 --no-cache-dir"
                                sh "python setup.py bdist_wheel"
                                sh """
                                    for f in dist/*.whl; do
                                        auditwheel repair --plat manylinux2014_x86_64 $f
                                    done
                                """
                            }
                        } 
                    }
                    stash name: "linux_wheel", includes: "dist/*"
                } } 
                stage("Build Arm Mac runtime") {
                    agent { label "macos && arm64 && xcode" }
                    steps {
                        setupRepo()
                        createZip("mac_arm")
                        extractRuntime()
                        buildXinterpreter()
                        dir("xformer") {
                            sh "curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-darwin-arm64"
                            sh "chmod +x bazelisk-darwin-arm64"
                            sh "./bazelisk-darwin-arm64 build //:xcore-opt --cpu=darwin_arm64 --copt=-fvisibility=hidden --copt=-mmacosx-version-min=11.0 --linkopt=-mmacosx-version-min=11.0 --linkopt=-dead_strip --//:disable_version_check"
                        }
                        createVenv("requirements.txt")
                        dir("python") { withVenv {
                            sh "pip install wheel setuptools setuptools-scm numpy six --no-cache-dir"
                            sh "python setup.py bdist_wheel --plat-name macosx_11_0_arm64"
                            stash name: "mac_arm_wheel", includes: "dist/*"
                        } }
                    }
                    post { cleanup { xcoreCleanSandbox() } }
                }
                // TODO: Windows build
            }
        }
        stage("Tests") { parallel {
            stage("Linux Test") {
                stages {
                    stage("Xformer Tests") { steps { withVenv {
                        sh "bazel --output_user_root=${env.BAZEL_USER_ROOT} test --remote_cache=${env.BAZEL_CACHE_URL} //Test:all --verbose_failures --test_output=errors --//:disable_version_check"
                    } } }
                    stage("Integration Tests") { steps { runTests("host") } }
                    stage("Notebook Tests") { steps { withVenv {
                        sh "pip install pytest nbmake"
                        sh "pytest --nbmake ./docs/notebooks/*.ipynb"
                        // TODO
                        // Test the pytorch to keras notebooks overnight? Need to manually install all requirements
                        // Also these train models so it takes a while
                    } } }
                }
            }
            stage("Host Arm Mac Test") {
                agent { label "mac && arm64" }
                stages {
                    stage("Setup") {
                        steps {
                            setupRepo()
                            createVenv("requirements.txt")
                            withVenv { sh "pip install -r requirements.txt" }
                        }
                    }
                    stage("Xformer Tests") {
                        steps {
                            sh "curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-darwin-arm64"
                            sh "chmod +x bazelisk-darwin-arm64"
                            sh "./bazelisk-darwin-arm64 --output_user_root=${env.BAZEL_USER_ROOT} test --remote_cache=${env.BAZEL_CACHE_URL} //Test:all --verbose_failures --test_output=errors --//:disable_version_check"
                        }
                    }
                    stage("Xinterpreter Tests") {
                        steps { runTests("mac_arm") }
                    }
                }
                post { cleanup { xcoreCleanSandbox() } }
            }
            stage("Device Test") {
                agent { label "xcore.ai-explorer && lpddr && !macos" }
                steps {
                    setupRepo()
                    createVenv("requirements.txt")
                    withVenv { sh "pip install -r requirements.txt" }
                    runTests("device")
                }
                post { cleanup { xcoreCleanSandbox() } }
            }
        } }
    }
    // TODO: Publish to PyPI if TAG_VERSION is set
    post { cleanup { xcoreCleanSandbox() } }
}


def createZip(String platform) {
    script {
        dir("xformer") { sh "./version_check.sh" }
        dir("third_party/lib_tflite_micro") {
            sh "mkdir -p build"
            dir("build") {
                if (platform == "device") {
                    sh "cmake .. --toolchain=../lib_tflite_micro/submodules/xmos_cmake_toolchain/xs3a.cmake"
                    sh "make create_zip -j4"
                } else if (platform == "linux" || platform == "mac_x86" || platform == "mac_arm") {
                    sh "cmake .. -DLIB_NAME=tflitemicro_${platform}"
                    sh "make create_zip -j4"
                } else if (platform == "windows") {
                    bat "cmake .. -DLIB_NAME=tflitemicro_${platform}"
                    // TODO: Make?
                }
            }
        }
    }
}


def setupRepo() {
    script {
        println "Stage running on: ${env.NODE_NAME}"
        checkout scm
        if (isUnix()){
            sh "git submodule update --init --recursive --jobs 4"
            sh "make -C third_party/lib_tflite_micro patch"
        } else {
            // bat "git submodule update --init --recursive --jobs 4"
            // dir("lib_tflite_micro/submodules/tflite-micro") {
            //     bat "git reset --hard && git apply --directory tensorflow ..\\..\\..\\patches\\tflite-micro.patch"
            // }
            // bat "cd lib_tflite_micro\\submodules\\tflite-micro && git reset --hard && git apply --directory tensorflow ..\\..\\..\\patches\\tflite-micro.patch"
            // bat "cd lib_tflite_micro && ..\\version_check.bat"
            // bat "mkdir build || echo 'Build directory already exists'"
            // bat "cd build && cmake .. && nmake /M:8"
        }
    }
}


def buildXinterpreter() {
    dir("python/xmos_ai_tools/xinterpreters/build") {
        sh "cmake .."
        sh "cmake --build . -t install --parallel 4 --config Release"
    }
}


def extractRuntime() {
    if (isUnix()) {
        sh "mv third_party/lib_tflite_micro/build/release_archive.zip python/xmos_ai_tools/runtime"
        dir("python/xmos_ai_tools/runtime") {
            sh "unzip release_archive.zip"
            sh "rm release_archive.zip"
            unstash "release_archive"
            sh "unzip release_archive.zip lib/libxtflitemicro.a -d ./"
        }
    } else {
        // TODO: Add device runtime
        bat "move third_party\\lib_tflite_micro\\build\\release_archive.zip python\\xmos_ai_tools\\runtime"
        dir("python\\xmos_ai_tools\\runtime") {
            bat "tar -xf release_archive.zip"
            bat "del release_archive.zip"
            unstash "release_archive"
            bat "tar -xf release_archive.zip lib\\libxtflitemicro.a -C .\\"
        }
    }
}


def runTests(String platform) {
    script {
        println "Stage running on: ${env.NODE_NAME}"
        withVenv {
            dir ("python") {
                if (platform == "linux" | platform == "device") {
                    unstash "linux_wheel"
                } else if (platform == "mac_arm") {
                    unstash "mac_arm_wheel"
                }
                sh "pip install dist/*"
            }
            if (platform == "device") {
                sh "pytest integration_tests/runner.py --models_path integration_tests/models/complex_models/non-bnns -n 1 --junitxml=integration_tests/integration_device_junit.xml"
                // lstms are always problematic
                sh "pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns/test_lstm -n 1"
                // test a float32 layer
                sh "pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns/test_detection_postprocess -n 1"
            } else {
                sh "pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns -n 8 --junitxml=integration_tests/integration_non_bnns_junit.xml"
                sh "pytest integration_tests/runner.py --models_path integration_tests/models/bnns --bnn -n 8 --junitxml=integration_tests/integration_bnns_junit.xml"
                sh "pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns --compiled -n 8 --junitxml=integration_compiled_non_bnns_junit.xml"
                sh "pytest integration_tests/runner.py --models_path integration_tests/models/bnns --bnn --compiled -n 8 --junitxml=integration_compiled_bnns_junit.xml"
                // notebook regression tests
            }
            junit "**/*_junit.xml"
        }
    }
}
