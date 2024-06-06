def configs = [
    ['arch': 'x86_64', 'os': 'ubuntu-22.04', 'build_type': 'Debug']
]

Map tasks = [failFast: false]

for (config in configs) {
    String nodeLabel = "${config.arch}-${config.os}"

    List configEnv = []
    configEnv << "BUILD_TYPE=${config.build_type}"

    tasks[nodeLabel] = { ->
        node(nodeLabel) {
            withEnv(configEnv) {
                withChecks(name: nodeLabel) {
                    stage("Checkout") {
                        dir("${WORKSPACE}/src/llvm-project") {
                            checkout changelog: false, poll: false, scm: scmGit(
                                branches: [[name: 'marco-llvm']],
                                extensions: [cloneOption(noTags: false, reference: '', shallow: false, timeout: 30)],
                                userRemoteConfigs: [[url: 'https://github.com/marco-compiler/llvm-project.git']]
                            )
                        }

                        dir("${WORKSPACE}/src/marco-runtime") {
                            git branch: 'master', url: 'https://github.com/marco-compiler/marco-runtime.git'
                        }

                        dir("${WORKSPACE}/src/marco") {
                            git branch: 'master', url: 'https://github.com/marco-compiler/marco.git'
                        }
                    }

                    stage('Virtual environment') {
                        sh '''#!/bin/bash
                        python3 -m venv .venv
                        source .venv/bin/activate
                        pip install --upgrade pip
                        pip install lit'''
                    }

                    stage("LLVM") {
                        stage("Configure") {
                            cmake arguments: "-S ${WORKSPACE}/src/llvm-project/llvm -B ${WORKSPACE}/build/llvm-project -G Ninja -DCMAKE_BUILD_TYPE=MinSizeRel -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_INSTALL_PREFIX=${WORKSPACE}/install/llvm-project -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DLLVM_TARGETS_TO_BUILD=\"host\" -DLLVM_INSTALL_UTILS=True -DLLVM_ENABLE_PROJECTS=\"clang;clang-tools-extra;mlir;openmp\" -DLLVM_PARALLEL_COMPILE_JOBS=${LLVM_PARALLEL_COMPILE_JOBS} -DLLVM_PARALLEL_LINK_JOBS=${LLVM_PARALLEL_LINK_JOBS}", installation: 'InSearchPath', label: 'Configure'
                        }

                        stage("Build") {
                            cmake arguments: "--build ${WORKSPACE}/build/llvm-project", installation: 'InSearchPath', label: 'Build'
                        }

                        stage("Install") {
                            cmake arguments: "--build ${WORKSPACE}/build/llvm-project --target install", installation: 'InSearchPath', label: 'Install'
                        }
                    }

                    stage('SUNDIALS') {
                        dir("${WORKSPACE}/build/sundials") {
                            sh "${WORKSPACE}/src/marco/dependencies/sundials.sh ${WORKSPACE}/install/sundials"
                        }
                    }

                    stage('MARCO Runtime') {
                        stage('Configure') {
                            cmake arguments: "-S ${WORKSPACE}/src/marco-runtime -B ${WORKSPACE}/build/marco-runtime -G Ninja -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${WORKSPACE}/install/marco-runtime -DLLVM_PATH=${WORKSPACE}/install/llvm-project -DSUNDIALS_PATH=${WORKSPACE}/install/sundials", installation: 'InSearchPath', label: 'Configure'
                        }

                        stage('Build') {
                            cmake arguments: "--build ${WORKSPACE}/build/marco-runtime", installation: 'InSearchPath', label: 'Build'
                        }

                        stage('Test') {
                            cmake arguments: "--build ${WORKSPACE}/build/marco-runtime --target test", installation: 'InSearchPath', label: 'Unit tests'
                        }

                        stage('Install') {
                            cmake arguments: "--build ${WORKSPACE}/build/marco-runtime --target install", installation: 'InSearchPath', label: 'Install'
                        }
                    }

                    stage('MARCO Compiler') {
                        stage('Configure') {
                            cmake arguments: "-S ${WORKSPACE}/src/marco -B ${WORKSPACE}/build/marco -G Ninja -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${WORKSPACE}/install/marco -DLLVM_PATH=${WORKSPACE}/install/llvm-project -DMARCORuntime_DIR=${WORKSPACE}/install/marco-runtime/lib/cmake/MARCORuntime -DLLVM_EXTERNAL_LIT=${WORKSPACE}/.venv/bin/lit", installation: 'InSearchPath', label: 'Configure'
                        }

                        stage('Build') {
                            cmake arguments: "--build ${WORKSPACE}/build/marco", installation: 'InSearchPath', label: 'Build'
                        }

                        stage('Test') {
                            cmake arguments: "--build ${WORKSPACE}/build/marco --target test", installation: 'InSearchPath', label: 'Unit tests'
                            cmake arguments: "--build ${WORKSPACE}/build/marco --target check", installation: 'InSearchPath', label: 'Regression tests'
                        }

                        stage('Install') {
                            cmake arguments: "--build ${WORKSPACE}/build/marco --target install", installation: 'InSearchPath', label: 'Install'
                        }
                    }
                }

                publishChecks(name: nodeLabel, conclusion: 'SUCCESS', summary: 'Check completed')
            }
        }
    }
}

stage("Parallel") {
    parallel(tasks)
}
