String configName = "debian-12"
String devDockerfile = "debian-12.Dockerfile"
String prodDockerfile = "prod-image.Dockerfile"

node {
    agent {
        label 'x86_64-linux'
    }

    String localWorkspace = "${WORKSPACE}/" + configName

    String srcPath = localWorkspace + "/src"
    String marcoSrcPath = srcPath + "/marco"

    stage("Checkout") {
        dir(marcoSrcPath) {
            def scmVars = checkout(scm)
            env.GIT_COMMIT = scmVars.GIT_COMMIT
        }
    }

    String dockerDevImageName = 'marco-compiler/marco-dev-release-' + configName + ":" + env.GIT_COMMIT[0..6]

    String dockerDevArgs =
        " --build-arg LLVM_PARALLEL_COMPILE_JOBS=${LLVM_PARALLEL_COMPILE_JOBS}" +
        " --build-arg LLVM_PARALLEL_LINK_JOBS=${LLVM_PARALLEL_LINK_JOBS}" +
        " --build-arg LLVM_BUILD_TYPE=Release" +
        " --build-arg LLVM_ENABLE_ASSERTIONS=OFF" +
        " --build-arg MARCO_RUNTIME_BUILD_TYPE=Release" +
        " -f " + marcoSrcPath + "/.jenkins/" + devDockerfile +
        " " + marcoSrcPath + "/.jenkins";

    String dockerProdImageName = 'marco-compiler/marco-prod-' + configName + ":" + env.GIT_COMMIT[0..6]

    String dockerProdArgs =
        " --build-arg BASE_IMAGE=" + dockerDevImageName +
        " --build-arg MARCO_COMMIT=" + env.GIT_COMMIT +
        " -f " + marcoSrcPath + "/.jenkins/" + prodDockerfile +
        " " + marcoSrcPath + "/.jenkins";

    stage('Build') {
        docker.build(dockerDevImageName, dockerDevArgs)
        dockerImage = docker.build(dockerProdImageName, dockerProdArgs)
    }

    docker.withRegistry('https://ghcr.io', 'marco-ci') {
        stage('Publish') {
            dockerImage.push()
            dockerImage.push("latest")
        }
    }
}
