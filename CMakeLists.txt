cmake_minimum_required(VERSION 3.20)

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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")

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

#------------------------------------------------------------------------------
# LLVM / MLIR configuration
#------------------------------------------------------------------------------

if (DEFINED LLVM_PATH)
	get_filename_component(LLVM_ABS_PATH ${LLVM_PATH} ABSOLUTE)
	set(LLVM_DIR "${LLVM_ABS_PATH}/lib/cmake/llvm")
	set(MLIR_DIR "${LLVM_ABS_PATH}/lib/cmake/mlir")
	set(Clang_DIR "${LLVM_ABS_PATH}/lib/cmake/clang")
endif()

find_package(Python3 REQUIRED)
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")

# These are a bit of a hack, because we're hijacking a lot of LLVM machinery.
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${Clang_DIR}")

include(HandleLLVMOptions)
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddClang)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
#add_definitions(${LLVM_DEFINITIONS})

# Search the tablegen executable
find_program(LLVM_TABLEGEN_EXE "llvm-tblgen"
	PATHS ${LLVM_TOOLS_BINARY_DIR}
	NO_DEFAULT_PATH
	NO_CMAKE_FIND_ROOT_PATH)

#------------------------------------------------------------------------------
# Runtime
#------------------------------------------------------------------------------

if (DEFINED MARCO_RUNTIME_PATH)
	get_filename_component(MARCO_RUNTIME_ABS_PATH ${MARCO_RUNTIME_PATH} ABSOLUTE)
	set(MARCORuntime_DIR "${MARCO_RUNTIME_ABS_PATH}/${CMAKE_INSTALL_LIBDIR}/cmake/MARCORuntime")
endif()

find_package(MARCORuntime CONFIG)

if (MARCORuntime_FOUND)
	message(STATUS "Using MARCORuntimeConfig.cmake in: ${MARCORuntime_DIR}")
else()
	message(STATUS "MARCO Runtime libraries not found")
endif()

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
		message(NOTICE "Shared libraries are not supported on Windows. Disabling them.")
		set(BUILD_SHARED_LIBS OFF)
	elseif (APPLE)
		message(NOTICE "Shared libraries are not supported on MacOS. Disabling them.")
		set(MARCO_BUILD_SHARED_LIBS OFF)
	endif()
endif()

if (MARCO_BUILD_SHARED_LIBS)
	if (NOT LLVM_LINK_LLVM_DYLIB)
		# If LLVM has been built statically, then shared libraries would lead
		# to duplication of frontend options.
		message(NOTICE "LLVM has not been built as shared library. Project will be built statically.")
		set(MARCO_BUILD_SHARED_LIBS OFF)
	elseif (ARG_DISABLE_LLVM_LINK_LLVM_DYLIB)
		# The situation is the same if the LLVM shared library has been built
		# but the LLVM infrastructure has been instructed not to link it.
		message(NOTICE "LLVM libraries are not linked to the LLVM shared library. Project will be built statically.")
		set(MARCO_BUILD_SHARED_LIBS OFF)
	endif()
endif()

if (MARCO_BUILD_SHARED_LIBS)
	set(MARCO_LIB_TYPE SHARED)
else()
	set(MARCO_LIB_TYPE STATIC)
endif()

# Add sanitizer
if (DEFINED MARCO_SANITIZER)
    message(STATUS "Using sanitizer: ${MARCO_SANITIZER}")
    add_compile_options("-fsanitize=${MARCO_SANITIZER}")
    add_link_options("-fsanitize=${MARCO_SANITIZER}")
endif()

# Add a compile-time definition for the version.
if (DEFINED MARCO_VERSION)
    add_compile_definitions(MARCO_VERSION="${MARCO_VERSION}")
else ()
    execute_process(
            COMMAND git rev-parse --short HEAD
            RESULT_VARIABLE MARCO_VERSION_RESULT
            OUTPUT_VARIABLE MARCO_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE)

    if (MARCO_VERSION_RESULT EQUAL "0")
        add_compile_definitions(MARCO_VERSION="${MARCO_VERSION}")
    else ()
        message(WARNING "Can't get commit hash. Error: ${MARCO_VERSION_RESULT}")
    endif ()
endif ()

# Paths
set(MARCO_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(MARCO_INCLUDE_DIR ${MARCO_ROOT}/include)

set(MARCO_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(MARCO_BUILD_INCLUDE_DIR ${MARCO_BUILD_DIR}/include)

set(MARCO_INCLUDE_DIRS
        ${MARCO_INCLUDE_DIR}
        ${MARCO_BUILD_INCLUDE_DIR})

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
include_directories(${CLANG_INCLUDE_DIRS})

# Add MARCO files to the include path
include_directories(${MARCO_INCLUDE_DIRS})

# Set the RPATH to include the paths to external libraries
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tool)

install(DIRECTORY ${MARCO_INCLUDE_DIR}/marco
		DESTINATION include
		COMPONENT marco-headers
		FILES_MATCHING
		PATTERN "*.h"
		PATTERN "*.inc"
		PATTERN "*.td")

install(DIRECTORY ${MARCO_BUILD_INCLUDE_DIR}/marco
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
option(MARCO_BUILD_MLIR_DOCS "Generate the Markdown MLIR documentation MARCO docs." ON)
option(MARCO_BUILD_DOXYGEN_DOCS "Generate the Doxygen documentation for MARCO." OFF)

add_custom_target(docs ALL)
add_subdirectory(docs)

#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------

# Search for the runtime library.
find_package(MARCORuntime CONFIG)

if (MARCORuntime_FOUND)
	set(MARCO_TEST_SIMULATION_ENABLED ON)
else()
	set(MARCO_TEST_SIMULATION_ENABLED OFF)
endif()

enable_testing()

FetchContent_Declare(
	googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.16.0)

set(INSTALL_GTEST OFF CACHE INTERNAL "")

# For Windows: prevent overriding the parent project's compiler / linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Define the default arguments to use with 'lit', and an option for the user to override
set(LIT_ARGS_DEFAULT "-sv --max-time 300")

if (MSVC_IDE OR XCODE)
	set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()

set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

add_subdirectory(test)
add_subdirectory(unittest)
