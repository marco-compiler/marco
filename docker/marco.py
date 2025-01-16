#!/usr/bin/env python3

import os
import shutil
import stat
import subprocess
import sys

DOCKER_SOCKET = "/var/run/docker.sock"
DOCKER_IMAGE = "ghcr.io/marco-compiler/marco-prod-debian-12"
INPUT_FILE_TYPES = (".mo", ".bmo", ".c", ".mlir", ".ll", ".o")

def main():
    # Make sure the Docker client is installed.
    if not shutil.which("docker"):
        print("Docker client not found")
        exit(1)

    # Make sure the Docker daemon is running.
    if not stat.S_ISSOCK(os.stat(DOCKER_SOCKET).st_mode):
        print("Docker daemon not running")
        exit(1)

    # Pull the Docker image.
    proc = subprocess.Popen(
        ["docker", "pull", DOCKER_IMAGE],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT)

    proc.wait()

    if proc.returncode != 0:
        for line in proc.stdout:
            print(line.decode("utf-8"), end = "")

        exit(proc.returncode)

    # Prepare the arguments for running MARCO inside a container.
    cmd = ["docker", "run", "--rm", "-it"]

    # Mount the directories containing any involved file.
    mounted_dirs = []

    for arg in sys.argv:

        if arg.endswith(INPUT_FILE_TYPES):
            file_path = os.path.abspath(arg)
            parent_dir_path = os.path.dirname(file_path)

            if not parent_dir_path in mounted_dirs:
                mounted_dirs += [parent_dir_path]
                cmd += ["-v", parent_dir_path + ":" + parent_dir_path]

    # Match user id and group id.
    cmd += ["-u", str(os.geteuid()) + ":" + str(os.getegid())]

    # Add the Docker image name.
    cmd += [DOCKER_IMAGE]

    # Forward all the original arguments.
    cmd += ["marco"]
    cmd += sys.argv[1:]

    # Run MARCO inside a container.
    proc = subprocess.Popen(cmd)
    proc.wait()
    exit(proc.returncode)

if __name__ == "__main__":
    main()
