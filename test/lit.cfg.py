# -*- Python -*-

import os
import sys
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'MARCO'

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.cpp', '.mo', '.mlir', '.test']

# test_format: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.marco_obj_root, 'test')

# On MacOS, set the environment variable for the path of the SDK to be used.
lit.util.usePlatformSdkOnDarwin(config, lit_config)

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
config.substitutions.append(('%runtime_lib_dir', config.marco_runtime_lib_dir))

if config.marco_runtime_found == "1":
    config.available_features.add("runtime-library")

if config.marco_runtime_ida_enabled == "ON":
    config.available_features.add("runtime-ida")

config.substitutions.append(("%sundials_lib_dir", config.marco_runtime_sundials_lib_dir))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])
llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# Tweak the PATH to include the tools dir.
path = os.path.pathsep.join((config.marco_tools_dir, config.llvm_tools_dir, config.environment['PATH']))
config.environment['PATH'] = path

tool_dirs = [
    config.marco_tools_dir,
    config.llvm_tools_dir
]

tools = [
    'mlir-cpu-runner',
    'marco',
    'modelica-opt'
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# Set the LD_LIBRARY_PATH
ld_library_path = os.path.pathsep.join((
    config.marco_libs_dir,
    config.llvm_libs_dir,
    config.environment.get('LD_LIBRARY_PATH','')))

config.environment['LD_LIBRARY_PATH'] = ld_library_path
