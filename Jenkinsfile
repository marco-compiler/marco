def configs = [
    ['name': 'debian-12', 'dockerfile': 'Debian.Dockerfile', 'args': '--build-arg VERSION=12'],
    ['name': 'ubuntu-22.04', 'dockerfile': 'Ubuntu.Dockerfile', 'args': '--build-arg VERSION=22.04'],
    ['name': 'fedora-40', 'dockerfile': 'Fedora.Dockerfile', 'args': '--build-arg VERSION=40']
]

Map tasks = [failFast: false]

for (config in configs) {
    String configName = "${config.name}"
    String dockerImage = "${config.image}"
    String dockerfile = "${config.dockerfile}"
    String configArgs = "${config.args}"

    List configEnv = []

    publishChecks(name: configName, status: 'QUEUED', summary: 'Queued')

    tasks[configName] = { ->
        node {
            String localWorkspace = "${WORKSPACE}/" + configName

            String srcPath = localWorkspace + "/src"
            String buildPath = localWorkspace + "/build"
            String installPath = localWorkspace + "/install"

            String marcoSrcPath = srcPath + "/marco"
            String marcoBuildPath = buildPath + "/marco"
            String marcoInstallPath = installPath + "/marco"

            stage("Checkout") {
                dir(marcoSrcPath) {
                    checkout(scm)
                }
            }

            String dockerMARCOImageName = 'marco-' + configName
            String llvmBuildType = "Release"
            String llvmEnableAssertions = "ON"

            String dockerArgs = configArgs +
                " --build-arg LLVM_BUILD_TYPE=" + llvmBuildType +
                " --build-arg LLVM_ENABLE_ASSERTIONS=" + llvmEnableAssertions +
                " --build-arg LLVM_PARALLEL_COMPILE_JOBS=${LLVM_PARALLEL_COMPILE_JOBS}" +
                " --build-arg LLVM_PARALLEL_LINK_JOBS=${LLVM_PARALLEL_LINK_JOBS}" +
                " -f " + marcoSrcPath + "/.jenkins/" + dockerfile +
                " " + marcoSrcPath + "/.jenkins";

            withEnv(configEnv) {
                publishChecks(name: configName, status: 'IN_PROGRESS', summary: 'In progress')

                docker.withRegistry('https://ghcr.io', 'github-app') {
                    def devImage = docker.build(
                        dockerMARCOImageName + '-dev',
                        "--build-arg MARCO_RUNTIME_BUILD_TYPE=Debug --build-arg LLVM_ENABLE_ASSERTIONS=ON " + dockerArgs)

                    devImage.push()

                    devImage.inside("-v $HOME/ccache:/ccache " + '-e "CCACHE_DIR=/ccache"') {
                        withChecks(name: configName) {
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

                            stage("Clean") {
                                dir(marcoBuildPath) {
                                    deleteDir()
                                }

                                dir(marcoInstallPath) {
                                    deleteDir()
                                }
                            }

                            stage('Configure') {
                                cmake arguments: "-S " + marcoSrcPath + " -B " + marcoBuildPath + " -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=" + marcoInstallPath + " -DLLVM_EXTERNAL_LIT=" + localWorkspace + "/.venv/bin/lit", installation: 'InSearchPath', label: 'Configure'
                            }

                            stage('Build') {
                                cmake arguments: "--build " + marcoBuildPath, installation: 'InSearchPath', label: 'Build'
                            }

                            stage('Test') {
                                cmake arguments: "--build " + marcoBuildPath + " --target test", installation: 'InSearchPath', label: 'Unit tests'
                                cmake arguments: "--build " + marcoBuildPath + " --target check", installation: 'InSearchPath', label: 'Regression tests'
                            }
                        }
                    }

                    publishChecks(name: configName, conclusion: 'SUCCESS', summary: 'Completed')
                }
            }
        }
    }
}

stage("Parallel") {
    parallel(tasks)
}
