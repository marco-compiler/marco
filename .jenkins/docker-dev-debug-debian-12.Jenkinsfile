String configName = "debian-12"
String dockerfile = "debian-12.Dockerfile"
String checkName = "docker-dev-image-debug"

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

    stage("Checkout") {
        dir(marcoSrcPath) {
            checkout(scm)
        }
    }

    String dockerMARCOImageName = 'marco-compiler/marco-dev-' + configName + "-debug"

    String dockerArgs =
        " --build-arg LLVM_PARALLEL_COMPILE_JOBS=${LLVM_PARALLEL_COMPILE_JOBS}" +
        " --build-arg LLVM_PARALLEL_LINK_JOBS=${LLVM_PARALLEL_LINK_JOBS}" +
        " --build-arg LLVM_BUILD_TYPE=Release" +
        " --build-arg LLVM_ENABLE_ASSERTIONS=ON" +
        " --build-arg MARCO_RUNTIME_BUILD_TYPE=Debug" +
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
