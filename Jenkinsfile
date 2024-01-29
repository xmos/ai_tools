@Library('xmos_jenkins_shared_library@v0.25.0') _

getApproval()

pipeline {
    agent none
    environment {
        REPO = "ai_tools"
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
        stage("Setup and build") { 
            agent { label "linux && x86_64 && !noAVX2" } 
            stages {
                stage("Setup") { steps {
                    println "Stage running on: ${env.NODE_NAME}"
                    checkout scm
                    sh "./build.sh -T init"
                    createVenv("requirements.txt")
                    withVenv { sh "pip install -r requirements.txt" }
                } }
                stage("Build") { steps { withVenv {
                    // build dll_interpreter for python interface
                    withTools(params.TOOLS_VERSION) {
                        sh "./build.sh -T xinterpreter -b"
                    }
                    // build xformer
                    dir("xformer") {
                        sh "wget https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-linux-amd64"
                        sh "chmod +x bazelisk-linux-amd64"
                        sh "./bazelisk-linux-amd64 --output_user_root=${env.BAZEL_USER_ROOT} build --remote_cache=${env.BAZEL_CACHE_URL} //:xcore-opt --verbose_failures --//:disable_version_check"
                        // TODO: factor this out
                        // yes, this is a test, but might as well not download bazel again
                        sh "./bazelisk-linux-amd64 --output_user_root=${env.BAZEL_USER_ROOT} test --remote_cache=${env.BAZEL_CACHE_URL} //Test:all --verbose_failures --test_output=errors --//:disable_version_check"
                    }
                    // build python wheel with xformer and install into env
                    dir ("python") {
                        sh "python3 setup.py bdist_wheel"
                        sh "pip install dist/*"
                        stash name:"xmos_ai_tools_wheel", includes: "dist/*"
                    }
                } } }
            }
            post { cleanup { xcoreCleanSandbox() } }
        } 
        stage("Tests") { parallel {
            stage("Host Test") {
                agent { label "linux && x86_64 && !noAVX2" }
                stages {
                    stage("Integration Tests") { steps { 
                        script { runTests("host") } 
                    } }
                    stage("Notebook Tests") { steps { withVenv {
                        sh "pip install pytest nbmake"
                        sh "pytest --nbmake ./docs/notebooks/*.ipynb"
                        // Test the pytorch to keras notebooks overnight? Need to manually install all requirements
                        // Also these train models so it takes a while
                        // sh "pytest --nbmake ./docs/notebooks/*.ipynb"
                    } } }
                }
                post { cleanup { xcoreCleanSandbox() } }
            }
            stage("Device Test") {
                agent { label "xcore.ai-explorer && lpddr && !macos" }
                steps { script { runTests("device") } }
                post { cleanup { xcoreCleanSandbox() } }
            }
        } }
    }
}

def runPytest(String test, String args) {
    timeout(time: 10, unit: 'MINUTES') {
        sh "xtagctl reset_all XCORE-AI-EXPLORER"
        sh "pytest integration_tests/runner.py --models_path integration_tests/models/${test} ${args} -s"
    }
}

def runTests(String platform) {
    println "Stage running on: ${env.NODE_NAME}"
    checkout scm
    sh "./build.sh -T init"
    createVenv("requirements.txt")
    withVenv {
        sh "pip install -r requirements.txt"
        dir ("python") {
            unstash "xmos_ai_tools_wheel"
            sh "pip install dist/*"
        }
        script {
            XMOS_AITOOLSLIB_PATH = sh(script: "python -c \"import xmos_ai_tools.runtime as rt; import os; print(os.path.dirname(rt.__file__))\"", returnStdout: true).trim()
            env.XMOS_AITOOLSLIB_PATH = XMOS_AITOOLSLIB_PATH
        }
        if (platform == "device") {
            sh "cd ${WORKSPACE} && git clone https://github0.xmos.com/xmos-int/xtagctl.git"
            sh "pip install -e ${WORKSPACE}/xtagctl"
            withTools(params.TOOLS_VERSION) {
                runPytest("complex_models/non-bnns/test_cnn_classifier", "-n 1 --tc 1 --device --junitxml=integration_tests/integration_device_1_junit.xml")
                runPytest("complex_models/non-bnns/test_cnn_classifier", "-n 1 --device --junitxml=integration_tests/integration_device_5_junit.xml")
                // lstms are always problematic
                runPytest("non-bnns/test_lstm", "-n 1 --tc 1 --device")
                runPytest("non-bnns/test_lstm", "-n 1 --device")
                runPytest("non-bnns/test_softmax", "-n 1 --device")
                // test a float32 layer
                runPytest("non-bnns/test_detection_postprocess", "-n 1 --device")
            }
        } else if (platform == "host") {
            sh "pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns -n 8 --junitxml=integration_tests/integration_non_bnns_1_junit.xml --tc 1"
            sh "pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns -n 8 --junitxml=integration_tests/integration_non_bnns_5_junit.xml"
            sh "pytest integration_tests/runner.py --models_path integration_tests/models/bnns --bnn -n 8 --junitxml=integration_tests/integration_bnns_junit.xml"
            sh "pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns --compiled -n 8 --junitxml=integration_compiled_non_bnns_junit.xml"
            sh "pytest integration_tests/runner.py --models_path integration_tests/models/bnns --bnn --compiled -n 8 --junitxml=integration_compiled_bnns_junit.xml"
            // notebook regression tests
        }
        junit "**/*_junit.xml"
    }
}
