# -*- Python -*-

import lit.util
import os
from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'MARCO'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.cpp', '.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.standalone_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

config.substitutions.append(('%llvm_include_dirs', config.llvm_include_dirs))
config.substitutions.append(('%runtime_h', os.path.join(config.standalone_src_root, 'lib/runtime/include')))
config.substitutions.append(('%libs', os.path.join(config.standalone_obj_root, 'lib')))
config.substitutions.append(('%runtime_lib', os.path.join(config.standalone_obj_root, 'lib', 'runtime', 'libMARCORuntime') + config.llvm_shlib_ext))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.standalone_obj_root, 'test')
config.standalone_tools_dir = os.path.join(config.standalone_obj_root, 'bin')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [config.standalone_tools_dir, config.llvm_tools_dir]
tools = [
    'mlir-cpu-runner',
    'marco',
    'modelica-opt'
]

llvm_config.add_tool_substitutions(tools, tool_dirs)