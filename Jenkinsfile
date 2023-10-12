@Library('xmos_jenkins_shared_library@v0.25.0') _

getApproval()

pipeline {
    agent none
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
            agent { label "linux && 64 && !noAVX2" } 
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
                    sh "./build.sh -T runtime-host -b"
                    sh "./build.sh -T xinterpreter-nozip -b"
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
                agent { label "linux && 64 && !noAVX2" }
                stages {
                    stage("Integration Tests") { steps { 
                        script { runTests("host") } 
                    } }
                    stage("Notebook Tests") { steps { withVenv {
                        sh "pip install pytest nbmake"
                        sh "pytest --nbmake ./docs/notebooks/**/*.ipynb"
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
        if (platform == "device") {
            sh "pytest integration_tests/runner.py --models_path integration_tests/models/complex_models/non-bnns -n 1 --junitxml=integration_tests/integration_device_junit.xml"
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
