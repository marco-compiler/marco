cmake_minimum_required(VERSION 3.9)

if (POLICY CMP0068)
	cmake_policy(SET CMP0068 NEW)
	set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if (POLICY CMP0075)
	cmake_policy(SET CMP0075 NEW)
endif()

if (POLICY CMP0077)
	cmake_policy(SET CMP0077 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Generate a CompilationDatabase (compile_commands.json file) for our build,
# for use by clang_complete, YouCompleteMe, etc.
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

#-------------------------------------------------------------------------------
# Project setup
#-------------------------------------------------------------------------------
project(marco LANGUAGES C CXX)

# Disable exceptions and RTTI
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -frtti")

# Set the default build type if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'Release' as none was specified")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"  "MinSizeRel" "RelWithDebInfo")
endif()

include(GNUInstallDirs)
include(FetchContent)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

#-------------------------------------------------------------------------------
# Sundials
#-------------------------------------------------------------------------------

option(MARCO_USE_BUILTIN_SUNDIALS "Use built-in Sundials (to be built separately, see Readme)" ON)

if (MARCO_USE_BUILTIN_SUNDIALS)
	set(SUNDIALS_DIR "${SUNDIALS_PATH}/lib/cmake/sundials")
endif()

find_package(SUNDIALS REQUIRED)

if (NOT MARCO_USE_BUILTIN_SUNDIALS)
  # Compute SUNDIALS's installation prefix
  get_filename_component(SUNDIALS_PATH "${SUNDIALS_DIR}" PATH)
  get_filename_component(SUNDIALS_PATH "${SUNDIALS_PATH}" PATH)
  get_filename_component(SUNDIALS_PATH "${SUNDIALS_PATH}" PATH)
  if(SUNDIALS_PATH STREQUAL "/")
    set(SUNDIALS_PATH "")
  endif()
endif()

set(SUNDIALS_INCLUDE_DIR ${SUNDIALS_PATH}/include)
set(SUNDIALS_LIBRARY_DIR ${SUNDIALS_PATH}/lib)

if (MARCO_USE_BUILTIN_SUNDIALS)
	link_directories("${SUNDIALS_LIBRARY_DIR}")
	install(DIRECTORY ${SUNDIALS_LIBRARY_DIR} DESTINATION dependencies)
endif()

#-------------------------------------------------------------------------------
# LLVM / MLIR configuration
#-------------------------------------------------------------------------------

if (DEFINED LLVM_PATH)
	# Emit a warning if both LLVM_ROOT and LLVM_DIR have been set
	if (DEFINED LLVM_DIR)
		message(WARNING "Both LLVM_PATH and LLVM_DIR have been set. LLVM_PATH has been ignored")
	else()
		set(LLVM_DIR "${LLVM_PATH}/lib/cmake/llvm")
	endif()

	# Emit a warning if both LLVM_ROOT and MLIR_DIR have been set
	if (DEFINED MLIR_DIR)
		message(WARNING "Both LLVM_PATH and MLIR_DIR have been set. LLVM_PATH has been ignored")
	else()
		set(MLIR_DIR "${LLVM_PATH}/lib/cmake/mlir")
	endif()
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# These are a bit of a hack, because we're hijacking alot of LLVM machinery
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(HandleLLVMOptions)
include(TableGen)
include(AddLLVM)
include(AddMLIR)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Search the tablegen executable
find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR})

#-------------------------------------------------------------------------------
# MARCO configuration
#-------------------------------------------------------------------------------

# Shared libraries
if (UNIX AND NOT APPLE)
	set(MARCO_BUILD_SHARED_LIBS_DEFAULT ON)
else()
	set(MARCO_BUILD_SHARED_LIBS_DEFAULT OFF)
endif()

option(MARCO_BUILD_SHARED_LIBS "Whether the MARCO libraries should be shared" ${MARCO_BUILD_SHARED_LIBS_DEFAULT})

if (MARCO_BUILD_SHARED_LIBS)
	if (WIN32)
		message(WARNING "Shared libraries are not supported on Windows. Disabling them.")
		set(BUILD_SHARED_LIBS OFF)
	elseif (APPLE)
		message(WARNING "Shared libraries are not supported on MacOS. Disabling them.")
		set(MARCO_BUILD_SHARED_LIBS OFF)
	endif()
endif()

if (MARCO_BUILD_SHARED_LIBS)
	if (NOT LLVM_LINK_LLVM_DYLIB)
		# If LLVM has been built statically, then shared libraries would lead
		# to duplication of frontend options.
		message(WARNING "LLVM has not been built as shared library. Project will be built statically.")
		set(MARCO_BUILD_SHARED_LIBS OFF)
	elseif (ARG_DISABLE_LLVM_LINK_LLVM_DYLIB)
		# The situation is the same if the LLVM shared library has been built
		# but the LLVM infrastructure has been instructed not to link it.
		message(WARNING "LLVM libraries are not linked to the LLVM shared library. Project will be built statically.")
		set(MARCO_BUILD_SHARED_LIBS OFF)
	endif()
endif()

if (MARCO_BUILD_SHARED_LIBS)
	set(MARCO_LIB_TYPE SHARED)
else()
	set(MARCO_LIB_TYPE STATIC)
endif()

# Paths
set(MARCO_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(MARCO_INCLUDE_PUBLIC_DIR ${MARCO_ROOT}/include/public)
set(MARCO_INCLUDE_PRIVATE_DIR ${MARCO_ROOT}/include/private)

set(MARCO_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(MARCO_BUILD_INCLUDE_PUBLIC_DIR ${MARCO_BUILD_DIR}/include/public)
set(MARCO_BUILD_INCLUDE_PRIVATE_DIR ${MARCO_BUILD_DIR}/include/private)

set(MARCO_INCLUDE_DIRS
		${MARCO_INCLUDE_PUBLIC_DIR}
		${MARCO_INCLUDE_PRIVATE_DIR}
		${MARCO_BUILD_INCLUDE_PUBLIC_DIR}
		${MARCO_BUILD_INCLUDE_PRIVATE_DIR})

set(MARCO_TOOLS_DIR ${MARCO_BUILD_DIR}/${CMAKE_INSTALL_BINDIR})
set(MARCO_LIBS_DIR ${MARCO_BUILD_DIR}/${CMAKE_INSTALL_LIBDIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(AddMARCO)

# Installing the headers and docs needs to depend on generating any public tablegen'd targets
add_custom_target(marco-headers)
set_target_properties(marco-headers PROPERTIES FOLDER "Misc")

# Add MLIR and LLVM headers to the include path
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Add MARCO files to the include path
include_directories(${MARCO_INCLUDE_DIRS})

# Set the RPATH to include the paths to external libraries
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#-------------------------------------------------------------------------------
# Simulation runtime profiling
#-------------------------------------------------------------------------------

option(MARCO_PROFILING "Generate code for runtime profiling." OFF)

if (MARCO_PROFILING)
	add_compile_definitions(MARCO_PROFILING)
endif()

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tool)

install(DIRECTORY ${MARCO_INCLUDE_PUBLIC_DIR}/marco
		DESTINATION include
		COMPONENT marco-headers
		FILES_MATCHING
		PATTERN "*.h"
		PATTERN "*.inc"
		PATTERN "*.td")

install(DIRECTORY ${MARCO_BUILD_INCLUDE_PUBLIC_DIR}/marco
		DESTINATION include
		COMPONENT marco-headers
		FILES_MATCHING
		PATTERN "*.h"
		PATTERN "*.inc"
		PATTERN "*.td"
		PATTERN "CMakeFiles" EXCLUDE)

add_subdirectory(cmake/modules)

#-------------------------------------------------------------------------------
# Documentation
#-------------------------------------------------------------------------------
add_custom_target(marco-docs)

option(MARCO_INCLUDE_DOCS "Generate build targets for the MARCO docs." ON)
option(MARCO_BUILD_DOCS "Generate the documentation for MARCO." OFF)

if (MARCO_BUILD_DOCS)
	add_custom_target(doxygen ALL)
endif()

if (MARCO_INCLUDE_DOCS)
	add_subdirectory(docs)
	add_dependencies(marco-docs mlir-doc)
endif()

#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------
enable_testing()

FetchContent_Declare(
	googletest
	URL https://github.com/google/googletest/archive/f7902802f1a61140e188223fb6d1c95925cbec4a.zip
)

set(INSTALL_GTEST OFF CACHE INTERNAL "")

# For Windows: prevent overriding the parent project's compiler / linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Define the default arguments to use with 'lit', and an option for the user to override
set(LIT_ARGS_DEFAULT "-sv")

if (MSVC_IDE OR XCODE)
	set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()

set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

add_subdirectory(test)
add_subdirectory(unittest)
