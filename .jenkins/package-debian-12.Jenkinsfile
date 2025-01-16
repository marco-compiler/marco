String configName = "debian-12"
String dockerfile = "debian-12.Dockerfile"
String checkName = "package-debian-12"

publishChecks(name: checkName, status: 'QUEUED', summary: 'Queued')

node {
    agent {
        label 'x86_64-linux'
    }

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

    String dockerMARCOImageName = 'marco-compiler/marco-package-' + configName

    String dockerArgs =
        " --build-arg LLVM_PARALLEL_COMPILE_JOBS=${LLVM_PARALLEL_COMPILE_JOBS}" +
        " --build-arg LLVM_PARALLEL_LINK_JOBS=${LLVM_PARALLEL_LINK_JOBS}" +
        " -f " + marcoSrcPath + "/.jenkins/" + dockerfile +
        " " + marcoSrcPath + "/.jenkins";

    publishChecks(name: checkName, status: 'IN_PROGRESS', summary: 'In progress')

    def dockerImage

    stage('Docker image') {
        dockerImage = docker.build(dockerMARCOImageName + ':latest', dockerArgs)
    }

    dockerImage.inside() {
        withChecks(name: checkName) {
            stage("OS information") {
                sh "cat /etc/os-release"
            }

            stage('Configure') {
                cmake arguments: "-S " + marcoSrcPath + " -B " + marcoBuildPath + " -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_LINKER_TYPE=MOLD -DCMAKE_INSTALL_PREFIX=" + marcoInstallPath, installation: 'InSearchPath', label: 'Configure'
            }

            stage('Install') {
                cmake arguments: "--build " + marcoBuildPath + " --target install", installation: 'InSearchPath', label: 'Install'
            }

            stage('Package') {
                sh "chmod +x " + marcoSrcPath + "/.jenkins/package/" + configName + "/build.sh"
                sh marcoSrcPath + "/.jenkins/package/" + configName + "/build.sh " + marcoSrcPath + " " + marcoInstallPath

                sshPublisher(publishers: [sshPublisherDesc(configName: 'marco-package', transfers: [sshTransfer(cleanRemote: false, excludes: '', execCommand: '', execTimeout: 120000, flatten: false, makeEmptyDirs: false, noDefaultExcludes: false, patternSeparator: '[, ]+', remoteDirectory: configName + "/amd64", remoteDirectorySDF: false, sourceFiles: '*.deb')], usePromotionTimestamp: false, useWorkspaceInPromotion: false, verbose: false)])
            }
        }
    }

    publishChecks(name: checkName, conclusion: 'SUCCESS', summary: 'Completed')
}
