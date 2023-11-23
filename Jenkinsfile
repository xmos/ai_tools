@Library('xmos_jenkins_shared_library@v0.25.0') _

getApproval()

pipeline {
    agent { label "linux && 64 && !noAVX2" } 
    environment {
        BAZEL_CACHE_URL = 'http://srv-bri-bld-cache:8080'
        BAZEL_USER_ROOT = "${WORKSPACE}/.bazel/"
        REBOOT_XTAG = '1'
    }
    parameters { // Available to modify on the job page within Jenkins if starting a build
        string( // use to try different tools versions
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
    stages {
        stage("Build device runtime") { steps {
                setupEnvironment()
                withVenv { createZip("device") }
                stash name: "release_archive", includes: "third_party/lib_tflite_micro/build/release_archive.zip"
        } }
        stage("Build host wheels") {
            parallel {
                stage("Build linux runtime") { steps { withVenv {
                    createZip("linux")
                    extractRuntime()
                    buildXinterpreter()
                    // build xformer
                    dir("xformer") {
                        sh "wget https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-linux-amd64"
                        sh "chmod +x bazelisk-linux-amd64"
                        sh "./bazelisk-linux-amd64 --output_user_root=${env.BAZEL_USER_ROOT} build --remote_cache=${env.BAZEL_CACHE_URL} //:xcore-opt --verbose_failures --//:disable_version_check"
                        // TODO: factor this out
                        // yes, this is a test, but might as well not download bazel again
                        sh "./bazelisk-linux-amd64 --output_user_root=${env.BAZEL_USER_ROOT} test --remote_cache=${env.BAZEL_CACHE_URL} //Test:all --verbose_failures --test_output=errors --//:disable_version_check"
                    }
                    // create wheel
                    dir ("python") {
                        sh "python3 setup.py bdist_wheel"
                        sh "pip install dist/*"
                        stash name: "linux_wheel", includes: "dist/*"
                    }
                } } }
                stage("Build x86 Mac runtime") {
                    agent { label "mac && x86_64" }
                    steps {
                        setupEnvironment()
                        withVenv {
                            createZip("mac_x86")
                        }
                    }
                    post { cleanup { xcoreCleanSandbox() } }
                }
                stage("Build Arm Mac runtime") {
                    agent { label "mac && arm64" }
                    steps {
                        setupEnvironment()
                        withVenv {
                            createZip("mac_arm")
                        }
                    }
                    post { cleanup { xcoreCleanSandbox() } }
                }
            }
        }
        stage("Tests") { parallel {
            stage("Host Test") {
                stages {
                    stage("Integration Tests") { steps { runTests("host") } }
                    stage("Notebook Tests") { steps { withVenv {
                        sh "pip install pytest nbmake"
                        sh "pytest --nbmake ./docs/notebooks/*.ipynb"
                        // Test the pytorch to keras notebooks overnight? Need to manually install all requirements
                        // Also these train models so it takes a while
                        // sh "pytest --nbmake ./docs/notebooks/*.ipynb"
                    } } }
                }
            }
            stage("Device Test") {
                agent { label "xcore.ai-explorer && lpddr && !macos" }
                steps {
                    setupEnvironment()
                    runTests("device")
                }
                post { cleanup { xcoreCleanSandbox() } }
            }
        } }
    }
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
                } else if (platform == "linux" || platform == "mac_x86" || platform == "mac_arm") {
                    sh "cmake .. -DLIB_NAME=tflitemicro_${platform}"
                } else if (platform == "windows") {
                    // Windows-specific cmake command
                }
                sh "make create_zip -j4"
            }
        }
    }
}


def setupEnvironment() {
    script {
        println "Stage running on: ${env.NODE_NAME}"
        checkout scm
        if (isUnix()){
            sh "git submodule update --init --recursive --jobs 4"
            sh "make -C third_party/lib_tflite_micro patch"
            createVenv("requirements.txt")
            withVenv { sh "pip install -r requirements.txt" }
        } else {
            // Windows specific setup steps
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
        bat "move third_party\\lib_tflite_micro\\build\\release_archive.zip python\\xmos_ai_tools\\runtime"
        dir("python\\xmos_ai_tools\\runtime") {
            bat "tar -xf release_archive.zip"
            bat "del release_archive.zip"
        }
    }
}


def runTests(String platform) {
    script {
        println "Stage running on: ${env.NODE_NAME}"
        withVenv {
            dir ("python") {
                unstash "linux_wheel"
                sh "pip install dist/*"
            }
            if (platform == "device") {
                sh "pytest integration_tests/runner.py --models_path integration_tests/models/complex_models/non-bnns -n 1 --junitxml=integration_tests/integration_device_junit.xml"
                // lstms are always problematic
                sh "pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns/test_lstm -n 1"
                // test a float32 layer
                sh "pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns/test_detection_postprocess -n 1"
            } else if (platform == "host") {
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
