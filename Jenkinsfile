def configs = [
    ['name': 'debian-12', 'os': 'debian', 'path': '.jenkins/Debian', 'args': '--build-arg VERSION=12'],
    ['name': 'ubuntu-22.04', 'os': 'ubuntu', 'path': '.jenkins/Ubuntu', 'args': '--build-arg VERSION=22.04'],
    ['name': 'fedora-40', 'os': 'fedora', 'path': '.jenkins/Fedora', 'args': '--build-arg VERSION=40']
]

Map tasks = [failFast: false]

for (config in configs) {
    String name = "${config.name}"
    String dockerImage = "${config.image}"
    String os = "${config.os}"
    String configPath = "${config.path}"
    String configArgs = "${config.args}"

    List configEnv = []
    configEnv << "BUILD_TYPE=${config.build_type}"

    publishChecks(name: name, status: 'QUEUED', summary: 'Queued')

    tasks[name] = { ->
        node {
            String localWorkspace = "${WORKSPACE}/" + name
            String srcPath = "${WORKSPACE}"

            String buildPath = localWorkspace + "/build"
            String installPath = localWorkspace + "/install"

            String llvmSrcPath = srcPath + "/llvm-project"
            String llvmBuildPath = buildPath + "/llvm-project"
            String llvmInstallPath = installPath + "/llvm-project"

            String sundialsBuildPath = buildPath + "/sundials"
            String sundialsInstallPath = installPath + "/sundials"

            String marcoRuntimeSrcPath = srcPath + "/marco-runtime"
            String marcoRuntimeBuildPath = buildPath + "/marco-runtime"
            String marcoRuntimeInstallPath = installPath + "/marco-runtime"

            String marcoSrcPath = srcPath + "/marco"
            String marcoBuildPath = buildPath + "/marco"
            String marcoInstallPath = installPath + "/marco"

            withEnv(configEnv) {
                stage("Checkout") {
                    dir(llvmSrcPath) {
                        checkout changelog: false, poll: false, scm: scmGit(
                            branches: [[name: 'marco-llvm']],
                            extensions: [cloneOption(noTags: false, reference: '', shallow: false, timeout: 30)],
                            userRemoteConfigs: [[url: 'https://github.com/marco-compiler/llvm-project.git']]
                        )
                    }

                    dir(marcoRuntimeSrcPath) {
                        git branch: 'master', url: 'https://github.com/marco-compiler/marco-runtime.git'
                    }

                    dir(marcoSrcPath) {
                        checkout(scm)
                    }
                }

                def image = docker.build('marco-' + name, configArgs + " " + marcoSrcPath + "/" + configPath)

                image.inside("-v $HOME/ccache:/ccache " + '-e "CCACHE_DIR=/ccache"') {
                    withChecks(name: name) {
                        publishChecks(name: name, status: 'IN_PROGRESS', summary: 'In progress')

                        stage("OS information") {
                            sh "cat /etc/os-release"
                        }

                       stage('Virtual environment') {
                            dir(localWorkspace) {
                                sh '''#!/bin/bash
                                python3 -m venv .venv
                                source .venv/bin/activate
                                pip install --upgrade pip
                                pip install lit'''
                            }
                        }

                        stage("LLVM") {
                            cmake arguments: "-S " + llvmSrcPath + "/llvm -B " + llvmBuildPath + " -G Ninja -DCMAKE_BUILD_TYPE=MinSizeRel -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_INSTALL_PREFIX=" + llvmInstallPath + " -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DLLVM_TARGETS_TO_BUILD=\"host\" -DLLVM_INSTALL_UTILS=True -DLLVM_ENABLE_PROJECTS=\"clang;clang-tools-extra;mlir;openmp\" -DLLVM_PARALLEL_COMPILE_JOBS=${LLVM_PARALLEL_COMPILE_JOBS} -DLLVM_PARALLEL_LINK_JOBS=${LLVM_PARALLEL_LINK_JOBS}", installation: 'InSearchPath', label: 'Configure'
                            cmake arguments: "--build " + llvmBuildPath, installation: 'InSearchPath', label: 'Build'
                            cmake arguments: "--build " + llvmBuildPath + " --target install", installation: 'InSearchPath', label: 'Install'
                        }

                        stage('SUNDIALS') {
                            dir(sundialsBuildPath) {
                                sh marcoSrcPath + "/dependencies/sundials.sh " + sundialsInstallPath
                            }
                        }

                        stage('MARCO Runtime') {
                            cmake arguments: "-S " + marcoRuntimeSrcPath + " -B " + marcoRuntimeBuildPath + " -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=" + marcoRuntimeInstallPath + " -DLLVM_PATH=" + llvmInstallPath + " -DSUNDIALS_PATH=" + sundialsInstallPath, installation: 'InSearchPath', label: 'Configure'
                            cmake arguments: "--build " + marcoRuntimeBuildPath, installation: 'InSearchPath', label: 'Build'
                            cmake arguments: "--build " + marcoRuntimeBuildPath + " --target test", installation: 'InSearchPath', label: 'Unit tests'
                            cmake arguments: "--build " + marcoRuntimeBuildPath + " --target install", installation: 'InSearchPath', label: 'Install'
                        }

                        stage('MARCO Compiler') {
                            cmake arguments: "-S " + marcoSrcPath + " -B " + marcoBuildPath + " -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=" + marcoInstallPath + " -DLLVM_PATH=" + llvmInstallPath + " -DMARCO_RUNTIME_PATH=" + marcoRuntimeInstallPath + " -DLLVM_EXTERNAL_LIT=" + localWorkspace + "/.venv/bin/lit", installation: 'InSearchPath', label: 'Configure'
                            cmake arguments: "--build " + marcoBuildPath, installation: 'InSearchPath', label: 'Build'
                            cmake arguments: "--build " + marcoBuildPath + " --target test", installation: 'InSearchPath', label: 'Unit tests'
                            cmake arguments: "--build " + marcoBuildPath + " --target check", installation: 'InSearchPath', label: 'Regression tests'
                            cmake arguments: "--build " + marcoBuildPath + " --target install", installation: 'InSearchPath', label: 'Install'
                        }
                    }
                }

                publishChecks(name: name, conclusion: 'SUCCESS', summary: 'Completed')
            }
        }
    }
}

stage("Parallel") {
    parallel(tasks)
}
