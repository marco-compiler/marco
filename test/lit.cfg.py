# -*- Python -*-

import os
import sys

import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# Name of the test suite.
config.name = 'MARCO'

# List of file extensions to treat as test files.
config.suffixes = [
    ".cpp",
    ".mo",
    ".mlir",
    ".test"
]

# Test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# Root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# Root path where tests should be run.
config.test_exec_root = os.path.join(config.marco_obj_root, 'test')

# On MacOS, set the environment variable for the path of the SDK to be used.
lit.util.usePlatformSdkOnDarwin(config, lit_config)

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

# Set the path containing the runtime library.
config.substitutions.append(('%runtime_lib_dir', config.marco_runtime_lib_dir))

# Check if the runtime library has been found.
# If not, the simulations can't be run.
if config.marco_runtime_found == "1":
    config.available_features.add("runtime-library")

# Check if the IDA solver is enabled.
if config.marco_runtime_ida_enabled == "ON":
    config.available_features.add("runtime-ida")

# Copy system environment.
llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"], append_path=True)
llvm_config.use_default_substitutions()

# List of directories to exclude from the testsuite.
config.excludes = [
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
    "lit.cfg.py",
    "lit.site.cfg.py"
]

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.marco_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [
    config.marco_tools_dir,
    config.llvm_tools_dir
]

tools = [
    "marco",
    "modelica-opt"
]

# Search for OMC.
if lit.util.which("omc") != None:
    config.available_features.add("omc")

llvm_config.add_tool_substitutions(tools, tool_dirs)

# Set the LD_LIBRARY_PATH
ld_library_path = os.path.pathsep.join((
    config.marco_libs_dir,
    config.llvm_libs_dir,
    config.environment.get("LD_LIBRARY_PATH", "")))

config.environment["LD_LIBRARY_PATH"] = ld_library_path
