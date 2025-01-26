String configName = "debian-12"
String dockerfile = "debian-12.Dockerfile"

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
            def scmVars = checkout(scm)
            env.GIT_COMMIT = scmVars.GIT_COMMIT
            tag = sh(script: 'git describe --tags --abbrev=0', returnStdout: true).trim()
        }
    }

    String dockerMARCOImageName = 'marco-compiler/marco-dev-debug-' + configName

    String dockerArgs =
        " --build-arg LLVM_PARALLEL_COMPILE_JOBS=${LLVM_PARALLEL_COMPILE_JOBS}" +
        " --build-arg LLVM_PARALLEL_LINK_JOBS=${LLVM_PARALLEL_LINK_JOBS}" +
        " --build-arg LLVM_BUILD_TYPE=Release" +
        " --build-arg LLVM_ENABLE_ASSERTIONS=ON" +
        " --build-arg MARCO_RUNTIME_BUILD_TYPE=Debug" +
        " -f " + marcoSrcPath + "/.jenkins/" + dockerfile +
        " " + marcoSrcPath + "/.jenkins";

    stage('Build') {
        dockerImage = docker.build(dockerMARCOImageName + ":" + env.GIT_COMMIT[0..6], dockerArgs)
    }

    docker.withRegistry('https://ghcr.io', 'marco-ci') {
        stage('Publish') {
            dockerImage.push()

            if (PUBLISH_VERSION_TAG && tag != "") {
                dockerImage.push(tag)
            }

            dockerImage.push("latest")
        }
    }
}
