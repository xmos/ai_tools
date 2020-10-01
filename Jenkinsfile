@Library('xmos_jenkins_shared_library@v0.14.2') _

getApproval()

pipeline {
    agent none

    parameters { // Available to modify on the job page within Jenkins if starting a build
        string( // use to try different tools versions
            name: 'TOOLS_VERSION',
            defaultValue: '15.0.1',
            description: 'The tools version to build with (check /projects/tools/ReleasesTools/)'
        )
        booleanParam( // use to check results of rolling all conda deps forward
            name: 'UPDATE_ALL',
            defaultValue: false,
            description: 'Update all conda packages before building'
        )
        booleanParam( // use to force a rebuild of the docker image, auto if Dockerfile modified or first build
            name: 'PUSH_IMAGE',
            defaultValue: false,
            description: 'Rebuild and push a new docker image'
        )
    }

    options { // plenty of things could go here
        //buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
    }

    stages {
        stage ("Check for existing images") {
            agent {
                label 'docker'
            }
            steps {
                script {
                    env.IMAGE_EXISTS = true
                    env.IMAGE_TAG = GIT_BRANCH.replace('/', '-')
                    try {
                        sh "docker pull docker-repo.xmos.com/xmos/ai_tools:${IMAGE_TAG}"
                    } catch {
                        env.IMAGE_EXISTS = false
                    }
                }
            }
        }
        stage ("Build and Push Image") {
            // This builds the Dockerfile into an image and
            // uploads it to our private docker registry
            // note: This could also be moved into a separate repo and
            //       the image treated as a separate versioned artifact
            when {
                anyOf {
                    // Not yet completed successfully
                    expression { return env.IMAGE_EXISTS }
                    // Dockerfile updated
                    changeset 'Dockerfile'
                    // Manual parameter
                    expression { return params.PUSH_IMAGE }
                }
            }
            agent {
                label 'docker'
            }
            steps {
                script {
                    def image = docker.build('xmos/ai_tools')
                    docker.withRegistry('https://docker-repo.xmos.com', 'nexus') {
                        // always push to git branch (overwriting previous tags)
                        image.push(IMAGE_TAG)
                        if (GIT_BRANCH=='master') {
                            // most recent master build is then default image
                            image.push('latest')
                            // all master runs can be recreated with short git hash
                            image.push(GIT_COMMIT.take(7))
                        }
                    }
                }
            }
        }
        stage ("Pull and Use Image") {
            // Runs everything in a docker container with the job workspace bind mounted
            // and the uid/gid matching that of the server running docker
            agent {
                docker {
                    // grab latest image tagged with branch
                    image 'xmos/ai_tools:${IMAGE_TAG}'
                    registryUrl 'https://docker-repo.xmos.com'
                    alwaysPull true
                }
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
                                          shallow: true,
                                          parentCredentials: true,
                                          recursiveSubmodules: true],
                                         [$class: 'CleanCheckout']],
                            userRemoteConfigs: [[credentialsId: 'xmos-bot',
                                                 url: 'git@github.com:xmos/ai_tools']]
                        ])
                        // create venv
                        sh "conda env create -q -p ai_tools_venv -f environment.yml"
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
                stage("Build/Test") {
                    // due to the Makefile, we've combined build and test stages
                    steps {
                        // below is how we can activate the tools
                        sh """pushd /XMOS/tools/${params.TOOLS_VERSION}/XMOS/xTIMEcomposer/${params.TOOLS_VERSION} && . SetEnv && popd &&
                              conda run -p ai_tools_venv make ci"""
                        // Any call to pytest can be given the "--junitxml SOMETHING_junit.xml" option
                        // This step collects these files for display in Jenkins UI
                        junit "**/*_junit.xml"
                    }
                }
            }
            post {
                cleanup {
                    cleanWs()
                }
            }
        }
    }
}
