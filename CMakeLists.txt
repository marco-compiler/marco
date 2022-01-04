cmake_minimum_required(VERSION 3.9)

######################################################################
#                              Project                               #
######################################################################
project(marco VERSION 1.0 LANGUAGES CXX)

# Disable exceptions and RTTI
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")

# Set the default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'Release' as none was specified")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"  "MinSizeRel" "RelWithDebInfo")
endif()

######################################################################
#                              Testing                               #
######################################################################
include(CTest)
add_subdirectory(googletest)

######################################################################
#                        CLANG & LLVM & MLIR                         #
######################################################################
find_package(Clang REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${CLANG_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddClang)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${CLANG_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# inherit build options from the LLVM build we are linking against
include(HandleLLVMOptions)

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

######################################################################
#                              Profiling                             #
######################################################################

# Set the profiling system to default OFF
if(NOT DEFINED MARCO_PROFILING)
	set(MARCO_PROFILING OFF)
endif()

if (${MARCO_PROFILING} STREQUAL ON)
	# If the profiling system is enabled, the macro is propagated to the libraries
	# so that the required code is inserted.

	add_compile_definitions(MARCO_PROFILING)
endif()

######################################################################
#                              Subfolders                            #
######################################################################

include(marcoMacros.cmake)
include(extra.cmake)
add_subdirectory(tool)
add_subdirectory(lib)
add_subdirectory(test)

######################################################################
#                                Boost                               #
######################################################################

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.45.0 COMPONENTS graph exception REQUIRED)

add_library(boost INTERFACE)
target_include_directories(boost INTERFACE ${Boost_INCLUDE_DIRS})
target_compile_definitions(boost INTERFACE -DBOOST_NO_EXCEPTIONS)
target_compile_features(boost INTERFACE cxx_std_17)
