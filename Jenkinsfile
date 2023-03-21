@Library('xmos_jenkins_shared_library@v0.23.0') _

getApproval()

pipeline {
    agent none
    environment {
        BAZEL_CACHE_URL = 'http://srv-bri-bld-cache:8080'
    }
    parameters { // Available to modify on the job page within Jenkins if starting a build
        string( // use to try different tools versions
            name: 'TOOLS_VERSION',
            defaultValue: '15.2.1',
            description: 'The tools version to build with (check /projects/tools/ReleasesTools/)'
        )
        booleanParam( // use to check results of rolling all conda deps forward
            name: 'UPDATE_ALL',
            defaultValue: false,
            description: 'Update all conda packages before building'
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
        stage("Build") {
            agent {
                label "xcore.ai-explorer"
            }
            stages {
                stage("Setup") {
                    steps {
                        println "Stage running on: ${env.NODE_NAME}"
                        // clone
                        checkout scm
                        sh 'git submodule update --init --recursive --depth 1 --jobs \$(nproc)'
                        // create venv and install pip packages
                        createVenv("requirements.txt")
                        withVenv {
                            sh "pip install -r requirements.txt"
                        }
                    }
                }
                stage("Build") {
                    steps {
                        // below is how we can activate the tools, NOTE: xTIMEcomposer -> XTC at tools 15.0.5 and later
                        // sh """. /XMOS/tools/${params.TOOLS_VERSION}/XMOS/XTC/${params.TOOLS_VERSION}/SetEnv && //
                        // sh """. /XMOS/tools/${params.TOOLS_VERSION}/XMOS/XTC/${params.TOOLS_VERSION}/SetEnv &&
                        //       . activate ./ai_tools_venv &&
                        //       cd third_party/lib_tflite_micro &&
                        //       make build &&
                        //       cd ../.. &&
                        //       make clean &&
                        //       make build
                        // """
                        // sh """. activate ./ai_tools_venv && cd experimental/xformer &&
                        //       bazel build --remote_cache=${BAZEL_CACHE_URL} //:xcore-opt --verbose_failures --//:disable_version_check
                        // """
                        // sh """. activate ./ai_tools_venv &&
                        //       (cd python && python3 setup.py bdist_wheel) &&
                        //       pip install ./python/dist/* &&
                        //       pip install -r "./requirements.txt"
                        // """
                        withVenv {
                            withTools(params.TOOLS_VERSION) {
                                dir("third_party/lib_tflite_micro") {
                                    sh "make patch"
                                }
                                dir("third_party/aisrv/app_integration_tests") {
                                    sh "xmake -j8"
                                }
                                sh "pip install xmos-ai-tools --pre --upgrade"
                            }
                        }
                    }
                }
                stage("Test") {
                    steps {
                        // xformer2 unit tests
                        // sh """. activate ./ai_tools_venv && cd experimental/xformer &&
                        //       bazel test --remote_cache=${BAZEL_CACHE_URL} //Test:all --verbose_failures --test_output=errors --//:disable_version_check
                        // """
                        // xformer2 integration tests
                        withVenv {
                            sh "pytest integration_tests/runner.py --models_path integration_tests/models/non-bnns/test_add -n 8 --junitxml=integration_non_bnns_junit.xml"
                        }
                        // Any call to pytest can be given the "--junitxml SOMETHING_junit.xml" option
                        // This step collects these files for display in Jenkins UI
                        junit "**/*_junit.xml"
                // regression test for xmos_ai_tools juypiter notebooks
                        // sh """. activate ./ai_tools_venv &&
                        //     pip install ./python/
                        //     pip install pytest nbmake
                        //     pytest --nbmake ./docs/notebooks/*.ipynb
                        // """
                        withTools(params.TOOLS_VERSION) {
                            sh "xrun -l"
                        }
                    }
                }
            }
            post {
                cleanup {
                    xcoreCleanSandbox()
                }
            }
        }
    }
}
