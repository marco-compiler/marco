String configName = "debian-12"
String dockerfile = "prod-debian-12.Dockerfile"
String checkName = "docker-prod-image"

publishChecks(name: checkName, status: 'QUEUED', summary: 'Queued')

node {
    agent 'x86_64-ubuntu-22.04'

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

    String dockerMARCOImageName = 'marco-compiler/marco-prod-' + configName

    String dockerArgs =
        " --build-arg LLVM_PARALLEL_COMPILE_JOBS=${LLVM_PARALLEL_COMPILE_JOBS}" +
        " --build-arg LLVM_PARALLEL_LINK_JOBS=${LLVM_PARALLEL_LINK_JOBS}" +
        " -f " + marcoSrcPath + "/.jenkins/" + dockerfile +
        " " + marcoSrcPath + "/.jenkins";

    publishChecks(name: checkName, status: 'IN_PROGRESS', summary: 'In progress')

    def dockerImage

    stage('Build') {
        dockerImage = docker.build(dockerMARCOImageName + ':latest', dockerArgs)
    }

    docker.withRegistry('https://ghcr.io', 'marco-ci') {
        stage('Publish') {
            dockerImage.push()
        }
    }

    publishChecks(name: checkName, conclusion: 'SUCCESS', summary: 'Completed')
}
