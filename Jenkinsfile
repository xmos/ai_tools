@Library('xmos_jenkins_shared_library@v0.25.0') _

getApproval()

def runPytestDevice(String test, String args, String junit) {
    timeout(time: 60, unit: 'MINUTES') {
        sh "xtagctl reset_all XCORE-AI-EXPLORER"
        sh "pytest integration_tests/runner.py --models_path integration_tests/models/${test} ${args} --device --junitxml=integration_tests/integration_device_${junit}_junit.xml"
    }
}

def runPytestHost(String test, String args, String junit) {
    sh "pytest integration_tests/runner.py --models_path integration_tests/models/${test} ${args} --junitxml=integration_tests/integration_host_${junit}_junit.xml"
}

def dailyDeviceTest = {
    timeout(time: 20, unit: 'MINUTES') {
        sh "xtagctl reset_all XCORE-AI-EXPLORER"
        sh "pytest examples/app_mobilenetv2"
    }
    runPytestDevice("8x8/test_slice", "-n 1 --tc 1", "slice_1")
    runPytestDevice("8x8/test_lstm", "-n 1 --tc 1", "lstm_1")
    runPytestDevice("8x8/test_lstm", "-n 1", "lstm_5")
    runPytestDevice("complex_models/8x8/test_cnn_classifier", "-n 1 --tc 1", "cnn_classifier_1")
    runPytestDevice("complex_models/8x8/test_cnn_classifier", "-n 1", "cnn_classifier_5")
    runPytestDevice("8x8/test_softmax", "-n 1 --device", "softmax_5")
    runPytestDevice("8x8/test_detection_postprocess", "-n 1", "detection_postprocess_5")
    runPytestDevice("16x8/", "-n 1", "16x8_5")
}

def dailyHostTest = {
    runPytestHost("float32", "-n 8 --tc 1", "float32_1")
    runPytestHost("16x8", "-n 8 --tc 5", "16x8_5")
    runPytestHost("complex_models/8x8", "-n 2 --tc 1", "complex_8x8_5")
    runPytestHost("complex_models/float32", "-n 1 --tc 1", "complex_float32_5")
    runPytestHost("8x8", "-n 8 --tc 1", "8x8_1")
    runPytestHost("8x8", "-n 8", "8x8_5")
    runPytestHost("8x8", "--compiled -n 8", "compiled_8x8")
    runPytestHost("bnns", "--bnn -n 8", "bnns")
    runPytestHost("bnns", "--bnn --compiled -n 8", "compiled_bnns")
}

def runTests(String platform, Closure body) {
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
                body()
            }
        } else if (platform == "host") {
            body()
        }
        junit "**/*_junit.xml"
    }
}

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
            when { branch pattern: "PR-.*", comparator: "REGEXP" }
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
        stage("Tests") {
        when { branch pattern: "PR-.*", comparator: "REGEXP" }
            parallel {
                stage("Host Test") {
                    agent { label "linux && x86_64 && !noAVX2" }
                    stages {
                        stage("Integration Tests") { steps { 
                            script { runTests("host", dailyHostTest) } 
                        } }
                        stage("Notebook Tests") { steps { withVenv {
                            sh "pip install pytest nbmake"
                            sh "pytest --nbmake ./docs/notebooks/*.ipynb"
                        } } }
                    }
                    post { cleanup { xcoreCleanSandbox() } }
                }
                stage("Device Test") {
                    agent { label "xcore.ai-explorer && lpddr && !macos" }
                    steps { script { runTests("device", dailyDeviceTest) } }
                    post {
                        archiveArtifacts artifacts: 'examples/app_mobilenetv2/arena_sizes.csv', allowEmptyArchive: true
                        cleanup { xcoreCleanSandbox() }
                    }
                }
            }
        }
    }
}
