@Library('xmos_jenkins_shared_library@v0.14.2') _

getApproval()

pipeline {
    agent {
        dockerfile {
            args "-v /home/jenkins/.keras:/root/.keras -v /etc/passwd:/etc/passwd:ro"
        }
    }
    environment {
        BAZEL_CACHE_URL = 'http://srv-bri-bld-cache:8080'
    }
    parameters { // Available to modify on the job page within Jenkins if starting a build
        string( // use to try different tools versions
            name: 'TOOLS_VERSION',
            defaultValue: '15.0.6',
            description: 'The tools version to build with (check /projects/tools/ReleasesTools/)'
        )
        booleanParam( // use to check results of rolling all conda deps forward
            name: 'UPDATE_ALL',
            defaultValue: false,
            description: 'Update all conda packages before building'
        )
    }

    options { // plenty of things could go here
        timestamps()
    // on develop discard builds after a certain number else keep forever
        buildDiscarder(logRotator(
            numToKeepStr:         env.BRANCH_NAME ==~ /develop/ ? '100' : '',
            artifactNumToKeepStr: env.BRANCH_NAME ==~ /develop/ ? '100' : ''
        ))
    }

    stages {
        stage("Setup") {
            // Clone and install build dependencies
            steps {
                // clean auto default checkout
                sh "rm -rf *"
                // clone
                checkout([
                    $class: 'GitSCM',
                    branches: scm.branches,
                    doGenerateSubmoduleConfigurations: false,
                    extensions: [[$class: 'SubmoduleOption',
                                  threads: 8,
                                  timeout: 20,
                                  shallow: false,
                                  parentCredentials: true,
                                  recursiveSubmodules: true],
                                 [$class: 'CleanCheckout']],
                    userRemoteConfigs: [[credentialsId: 'xmos-bot',
                                         url: 'git@github.com:xmos/ai_tools']]
                ])
                // create venv and install pip packages
                sh "conda env create -q -p ai_tools_venv -f ./environment.yml"
                sh """. activate ./ai_tools_venv &&
                      pip install -r requirements.txt
                """
                // Install xmos tools version
                sh "/XMOS/get_tools.py " + params.TOOLS_VERSION
            }
        }
        stage("Update all packages") {
            // Roll all conda packages forward beyond their pinned versions
            when { expression { return params.UPDATE_ALL } }
            steps {
                sh "conda update --all -y -q -p ai_tools_venv"
            }
        }
        stage("Build") {
            steps {
                // below is how we can activate the tools, NOTE: xTIMEcomposer -> XTC at tools 15.0.5 and later
                // sh """. /XMOS/tools/${params.TOOLS_VERSION}/XMOS/XTC/${params.TOOLS_VERSION}/SetEnv && //
                sh """. /XMOS/tools/${params.TOOLS_VERSION}/XMOS/XTC/${params.TOOLS_VERSION}/SetEnv &&
                      . activate ./ai_tools_venv &&
                      cd third_party/lib_tflite_micro &&
                      make build &&
                      cd ../.. &&
                      make clean &&
                      make build
                """
                sh """. activate ./ai_tools_venv && cd experimental/xformer &&
                      bazel build --remote_cache=${BAZEL_CACHE_URL} //:xcore-opt --verbose_failures --//:disable_version_check
                """
                sh """. activate ./ai_tools_venv &&
                      (cd python && python3 setup.py bdist_wheel) &&
                      pip install ./python/dist/* &&
                      pip install -r "./requirements.txt"
                """
            }
        }
        stage("Test") {
            steps {
                // xformer2 unit tests
        sh """. activate ./ai_tools_venv && cd experimental/xformer &&
                      bazel test --remote_cache=${BAZEL_CACHE_URL} //Test:all --verbose_failures --test_output=errors --//:disable_version_check
                """
        // xformer2 integration tests
                sh """. activate ./ai_tools_venv &&
                      make test
                """
                // Any call to pytest can be given the "--junitxml SOMETHING_junit.xml" option
                // This step collects these files for display in Jenkins UI
                junit "**/*_junit.xml"
        // regression test for xmos_ai_tools juypiter notebooks
                sh """. activate ./ai_tools_venv &&
                    pip install ./python/
                    pip install pytest nbmake
                    pytest --nbmake ./docs/notebooks/*.ipynb
                """
            }
        }
    }
    post {
        cleanup {
            cleanWs()
        }
    }
}
