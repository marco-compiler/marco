cmake_minimum_required(VERSION 3.9)

if(POLICY CMP0068)
	cmake_policy(SET CMP0068 NEW)
	set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
	cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
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
project(marco LANGUAGES CXX)

# Disable exceptions and RTTI
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")

# Set the default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'Release' as none was specified")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"  "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

#-------------------------------------------------------------------------------
# Clang / MLIR / LLVM configuration
#-------------------------------------------------------------------------------

find_package(Clang REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# These are a bit of a hack, because we're hijacking alot of LLVM machinery
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${CLANG_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(HandleLLVMOptions)
include(TableGen)
include(AddClang)
include(AddLLVM)
include(AddMLIR)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Search the tablegen executable
find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR})

#-------------------------------------------------------------------------------
# MARCO configuration
#-------------------------------------------------------------------------------

set(MARCO_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(MARCO_INCLUDE_PUBLIC_DIR ${MARCO_ROOT}/include/public)
set(MARCO_INCLUDE_PRIVATE_DIR ${MARCO_ROOT}/include/private)

set(MARCO_BUILD_DIR ${PROJECT_BINARY_DIR})
set(MARCO_BUILD_INCLUDE_PUBLIC_DIR ${MARCO_BUILD_DIR}/include/public)
set(MARCO_BUILD_INCLUDE_PRIVATE_DIR ${MARCO_BUILD_DIR}/include/private)

set(MARCO_INCLUDE_DIRS
		${MARCO_INCLUDE_PUBLIC_DIR}
		${MARCO_INCLUDE_PRIVATE_DIR}
		${MARCO_BUILD_INCLUDE_PUBLIC_DIR}
		${MARCO_BUILD_INCLUDE_PRIVATE_DIR})

set(MARCO_TOOLS_DIR ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(AddMARCO)

# Installing the headers and docs needs to depend on generating any public tablegen'd targets
add_custom_target(marco-headers)
set_target_properties(marco-headers PROPERTIES FOLDER "Misc")
#add_custom_target(marco-doc)

# Add MLIR and LLVM headers to the include path
include_directories(${CLANG_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Add MARCO files to the include path
include_directories(${MARCO_INCLUDE_DIRS})

#-------------------------------------------------------------------------------
# MARCO profiling
#-------------------------------------------------------------------------------

# Set the profiling system to default OFF
if(NOT DEFINED MARCO_PROFILING)
	set(MARCO_PROFILING OFF)
endif()

if (${MARCO_PROFILING} STREQUAL ON)
	# If the profiling system is enabled, the macro is propagated to the libraries
	# so that the required code is inserted.

	add_compile_definitions(MARCO_PROFILING)
endif()

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tool)
add_subdirectory(unittest)

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
# Boost (will be removed soon)
#-------------------------------------------------------------------------------

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.45.0 COMPONENTS graph exception REQUIRED)

add_library(boost INTERFACE)
target_include_directories(boost INTERFACE ${Boost_INCLUDE_DIRS})
target_compile_definitions(boost INTERFACE -DBOOST_NO_EXCEPTIONS)
target_compile_features(boost INTERFACE cxx_std_17)
