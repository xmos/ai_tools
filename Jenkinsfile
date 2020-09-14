@Library('xmos_jenkins_shared_library@v0.14.1') _


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
                    changeset 'Dockerfile'
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
                        sh """#!/bin/bash -l
                              env"""
                        sh "conda run -n .venv python -c 'import tensorflow'"
                        sh """#!/bin/bash -l
                              xcc --version"""
                    }
                }
                stage("Build") {
                    steps {
                        sh """#!/bin/bash -l
                              conda run -n .venv make foobar"""
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
