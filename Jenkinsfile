@Library('xmos_jenkins_shared_library@v0.14.1') _


pipeline {
    agent none

    parameters {
        booleanParam(
            name: 'PUSH_IMAGE',
            defaultValue: false,
            description: 'Whether to rebuild and push a new docker image'
        )
    }

    stages {
        stage ("Build and Push Image") {
            when {
                anyOf {
                    changeset pattern: "(environment.*\\.yml)|(Dockerfile)", comparator: "REGEXP"
                    expression { return params.BUILD_IMAGE }
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
            steps {
                // we need a withConda or something
                sh "python -V"
                sh "conda list --export"
                sh """#!/bin/bash
                      conda init bash
                      conda activate ai_tools_venv
                      env > venv.env"""
                withEnv(parseEnvFile("venv/env")) {
                    sh "pip install --no-dependencies -e ./tflite2xcore"
                    sh "conda list --export"
                    sh "make all"
                }
            }
        }
    }
}
