#------------------------------------------------------------------------------
# Project setup
#------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.9)

project(marco-runtime LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

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

# CMake configuration files.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AddMARCORuntime)

#------------------------------------------------------------------------------
# CLI
#------------------------------------------------------------------------------

option(MARCO_ENABLE_CLI "Enable the command-line interface" ON)

if (MARCO_ENABLE_CLI)
    add_compile_definitions(CLI_ENABLE)
endif()

#------------------------------------------------------------------------------
# Multithreading
#------------------------------------------------------------------------------

# Find the thread library.
find_package(Threads)

if (Threads_FOUND)
    add_compile_definitions(THREADS_ENABLE)
endif()

#------------------------------------------------------------------------------
# LLVM / MLIR configuration
#------------------------------------------------------------------------------

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

# These are a bit of a hack, because we're hijacking a lot of LLVM machinery.
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

#------------------------------------------------------------------------------
# SUNDIALS libraries
#------------------------------------------------------------------------------

# Find the SUNDIALS libraries.
option(MARCO_ENABLE_SUNDIALS "Enable the usage of the SUNDIALS suite" ON)
option(MARCO_USE_BUILTIN_SUNDIALS "Use built-in SUNDIALS libraries (to be built separately, see README)" ON)

if (MARCO_ENABLE_SUNDIALS)
    if (MARCO_USE_BUILTIN_SUNDIALS)
        set(SUNDIALS_DIR "${SUNDIALS_PATH}/lib/cmake/sundials")
    endif()

    find_package(SUNDIALS)

    if (SUNDIALS_FOUND)
        set(IDA_ENABLE ON)
    else()
        set(IDA_ENABLE OFF)
    endif()

    if (NOT MARCO_USE_BUILTIN_SUNDIALS)
        # Compute SUNDIALS installation prefix
        get_filename_component(SUNDIALS_PATH "${SUNDIALS_DIR}" PATH)
        get_filename_component(SUNDIALS_PATH "${SUNDIALS_PATH}" PATH)
        get_filename_component(SUNDIALS_PATH "${SUNDIALS_PATH}" PATH)

        if (SUNDIALS_PATH STREQUAL "/")
            set(SUNDIALS_PATH "")
        endif()
    endif()

    set(SUNDIALS_INCLUDE_DIR ${SUNDIALS_PATH}/include)
    set(SUNDIALS_LIBRARY_DIR ${SUNDIALS_PATH}/lib)

    if (MARCO_USE_BUILTIN_SUNDIALS)
        link_directories("${SUNDIALS_LIBRARY_DIR}")
        install(DIRECTORY ${SUNDIALS_INCLUDE_DIR} DESTINATION dependencies/sundials)
        install(DIRECTORY ${SUNDIALS_LIBRARY_DIR} DESTINATION dependencies/sundials)
    endif()

    list(APPEND DEPENDENCIES_LIB_NAMES
            sundials_ida
            sundials_kinsol
            sundials_nvecserial
            sundials_sunlinsolklu
            klu)
endif()

#------------------------------------------------------------------------------
# Runtime libraries
#------------------------------------------------------------------------------

# Profiling.
option(MARCO_PROFILING "Generate code for runtime profiling." OFF)

# Include directories.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${MLIR_INCLUDE_DIRS})

# Set the RPATH to include the paths to external libraries
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Add a target that depends on all the libraries, in order to ease the
# declaration of tests.
add_custom_target(marco-runtime)

# Discover the libraries.
add_subdirectory(lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/marco
        DESTINATION include
        COMPONENT marco-headers
        FILES_MATCHING
        PATTERN "*.h")

#------------------------------------------------------------------------------
# Configuration export
#------------------------------------------------------------------------------

install(EXPORT MARCORuntimeTargets
        FILE MARCORuntimeTargets.cmake
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
        NAMESPACE MARCORuntime::)

add_subdirectory(cmake/modules)

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

enable_testing()

if (NOT CMAKE_CROSSCOMPILING)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/f7902802f1a61140e188223fb6d1c95925cbec4a.zip
    )

    set(INSTALL_GTEST OFF CACHE INTERNAL "")

    # For Windows: prevent overriding the parent project's compiler / linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_subdirectory(unittest)
endif()
