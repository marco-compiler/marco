String configName = "ubuntu-22.04"
String dockerfile = "ubuntu-22.04.Dockerfile"

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
            def scmVars = checkout(scm)
            env.GIT_COMMIT = scmVars.GIT_COMMIT
        }
    }

    String dockerMARCOImageName = 'marco-compiler/marco-dev-' + configName

    String dockerArgs =
        " --build-arg LLVM_PARALLEL_COMPILE_JOBS=${LLVM_PARALLEL_COMPILE_JOBS}" +
        " --build-arg LLVM_PARALLEL_LINK_JOBS=${LLVM_PARALLEL_LINK_JOBS}" +
        " --build-arg LLVM_BUILD_TYPE=Release" +
        " --build-arg LLVM_ENABLE_ASSERTIONS=ON" +
        " --build-arg MARCO_RUNTIME_BUILD_TYPE=Debug" +
        " -f " + marcoSrcPath + "/.jenkins/" + dockerfile +
        " " + marcoSrcPath + "/.jenkins";

    stage("Docker image") {
        dockerImage = docker.build(dockerMARCOImageName + ":" + env.GIT_COMMIT[0..6], dockerArgs)
    }

    dockerImage.inside() {
        stage("OS information") {
            sh "cat /etc/os-release"
        }

        stage('Configure') {
            cmake arguments: "-S " + marcoSrcPath + " -B " + marcoBuildPath + " -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_LINKER_TYPE=MOLD -DCMAKE_INSTALL_PREFIX=" + marcoInstallPath + " -DPython3_EXECUTABLE=/virtualenv/bin/python -DMARCO_SANITIZER=address -DCMAKE_EXPORT_COMPILE_COMMANDS=ON", installation: 'InSearchPath', label: 'Configure'
        }

        stage('Build') {
            cmake arguments: "--build " + marcoBuildPath, installation: 'InSearchPath', label: 'Build'
        }

        stage('Unit test') {
            cmake arguments: "--build " + marcoBuildPath + " --target test", installation: 'InSearchPath', label: 'Unit tests'
        }

        stage('Regression test') {
            cmake arguments: "--build " + marcoBuildPath + " --target check", installation: 'InSearchPath', label: 'Regression tests'
        }

        stage('Install') {
            cmake arguments: "--build " + marcoBuildPath + " --target install", installation: 'InSearchPath', label: 'Install'
        }
    }
}
