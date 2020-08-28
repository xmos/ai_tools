@Library('xmos_jenkins_shared_library@feature/view_env_path') _


pipeline {
    agent none

    parameters {
        booleanParam(
            name: 'UPDATE_ALL',
            defaultValue: false,
            description: 'Update all conda packages before building'
        )
        booleanParam(
            name: 'PUSH_IMAGE',
            defaultValue: false,
            description: 'Rebuild and push a new docker image'
        )
    }

    stages {
        stage ("Build and Push Image") {
            when {
                anyOf {
                    changeset "Dockerfile"
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
                        image.push(GIT_BRANCH)
                    }
                }
            }
        }
        stage ("Pull and Use Image") {
            agent {
                docker {
                    image 'xmos/ai_tools:${GIT_BRANCH}'
                    registryUrl 'https://docker-repo.xmos.com'
                    alwaysPull true
                }
            }
            stages {
                stage("Create env") {
                    steps {
                        sh "conda env create -n .venv -f environment.yml"
                        sh "conda list"
                        sh "conda list -n .venv"
                        withEnv(["CONDA_DEFAULT_ENV=.venv"]) {
                            sh "conda list"
                        }
                    }
                }
                stage("Update all packages") {
                    when { expression { return params.UPDATE_ALL } }
                    steps {
                        sh "conda update --all -n .venv"
                    }
                }
                stage("Check") {
                    steps {
                        viewEnv("/XMOS/xTIMEcomposer/${env.TOOLS_VERSION}") {
                            sh "conda run -n .venv python -c 'import tensorflow'"
                            sh "xcc --version"
                        }
                    }
                }
                stage("Build") {
                    steps {
                        viewEnv('/XMOS/xTIMEcomposer/${TOOLS_VERSION}') {
                            sh "conda run -n .venv make all"
                        }
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
