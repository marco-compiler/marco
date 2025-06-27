#!/usr/bin/env python3

import os
import shutil
import stat
import subprocess
import sys

DOCKER_SOCKET = "/var/run/docker.sock"
DOCKER_IMAGE = "marco-local-3"
INPUT_FILE_TYPES = (".mo", ".bmo", ".c", ".mlir", ".ll", ".o", ".so")

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
    #proc = subprocess.Popen(
     #   ["docker", "pull", DOCKER_IMAGE],
      #  stdout = subprocess.PIPE,
       # stderr = subprocess.STDOUT)

    #proc.wait()

    #for line in proc.stdout:
     #   print(line.decode("utf-8"), end = "")

    #if proc.returncode != 0:
      #  exit(proc.returncode)

    # Prepare the arguments for running MARCO inside a container.
    cmd = ["docker", "run", "--rm", "-it"]

   #sys.argv += ["-I " + header_dir, "-L" + lib_dir, "-lmialib"]

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

    # Set the working directory.
    cmd += ["-w", os.getcwd()]

    # Add the Docker image name.
    cmd += [DOCKER_IMAGE]

    # Forward all the original arguments.
  #  cmd += ["-Xmarco","myLib.so"]
    #
    cmd += ["marco"]



    
    cmd += sys.argv[1:]
    cmd += ["library.o"]

 #   cmd += ["-Wl,--whole-archive"]
  #  cmd += ["-Wl,--shared"]
  #  cmd += ["-Wl,-L."]
   # cmd += ["-Wl,-R."]
   # cmd += ["-Wl,-I."]
 #   cmd += ["-Wl,-rpath=."]
#    cmd += ["-Wl,-shared"]

   # cmd += ["-Wl,-u discreteLog"]
  #   cmd += ["-Wl,-F discreteLog"]
  #  cmd += ["-Wl,-no-dynamic-linker"]





   # 

    print(cmd);
    # Run MARCO inside a container.
    print("Compiling with MARCO inside a container...")
    proc = subprocess.Popen(cmd)
    proc.wait()
    exit(proc.returncode)

if __name__ == "__main__":
    main()