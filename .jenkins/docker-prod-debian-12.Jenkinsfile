String configName = "debian-12"
String dockerfile = "debian-12.Dockerfile"
String checkName = "docker-prod-image"

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

    stage("Checkout") {
        dir(marcoSrcPath) {
            checkout(scm)
        }
    }

    String dockerMARCOImageName = 'marco-compiler/marco-prod-' + configName

    String dockerArgs =
        " --build-arg LLVM_PARALLEL_COMPILE_JOBS=${LLVM_PARALLEL_COMPILE_JOBS}" +
        " --build-arg LLVM_PARALLEL_LINK_JOBS=${LLVM_PARALLEL_LINK_JOBS}" +
        " --build-arg LLVM_BUILD_TYPE=Release" +
        " --build-arg LLVM_ENABLE_ASSERTIONS=OFF" +
        " --build-arg MARCO_RUNTIME_BUILD_TYPE=Release" +
        " -f " + marcoSrcPath + "/.jenkins/" + dockerfile +
        " " + marcoSrcPath + "/.jenkins";

    publishChecks(name: checkName, status: 'IN_PROGRESS', summary: 'In progress')

    def dockerImage

    stage('Build') {
        dockerImage = docker.build(dockerMARCOImageName + ':latest', dockerArgs)
    }

    docker.withRegistry('https://ghcr.io', 'marco-ci') {
        stage('Configure') {
            cmake arguments: "-S " + marcoSrcPath + " -B " + marcoBuildPath + " -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_LINKER_TYPE=MOLD -DCMAKE_INSTALL_PREFIX=" + marcoInstallPath, installation: 'InSearchPath', label: 'Configure'
        }

        stage('Install') {
            cmake arguments: "--build " + marcoBuildPath + " --target install", installation: 'InSearchPath', label: 'Install'
        }

        stage('Clean') {
            sh "rm -rf " + marcoSrcPath
            sh "rm -rf " + marcoBuildPath
        }

        stage('Publish') {
            dockerImage.push()
        }
    }

    publishChecks(name: checkName, conclusion: 'SUCCESS', summary: 'Completed')
}
