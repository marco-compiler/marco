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

    String marcoSrcPath = srcPath + "/marco"
    String marcoBuildPath = buildPath + "/marco"

    stage("Checkout") {
        dir(marcoSrcPath) {
            def scmVars = checkout(scm)
            env.GIT_COMMIT = scmVars.GIT_COMMIT
            tag = sh(script: 'git describe --tags --abbrev=0', returnStdout: true).trim()
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
        dockerImage = docker.build(dockerMARCOImageName + ":" + env.GIT_COMMIT[0..6], dockerArgs)
    }

    docker.withRegistry('https://ghcr.io', 'marco-ci') {
        stage('Configure') {
            cmake arguments: "-S " + marcoSrcPath + " -B " + marcoBuildPath + " -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_LINKER_TYPE=MOLD", installation: 'InSearchPath', label: 'Configure'
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

            if (tag != "") {
                dockerImage.push(tag)
            }

            dockerImage.push("latest")
        }
    }

    publishChecks(name: checkName, conclusion: 'SUCCESS', summary: 'Completed')
}
