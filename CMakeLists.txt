cmake_minimum_required(VERSION 3.9)

##############################
###        googletest      ###
##############################
SET(BUILD_GMOCK ON CACHE INTERNAL "")
SET(BUILD_GTEST ON CACHE INTERNAL "")
SET(INSTALL_GMOCK OFF CACHE INTERNAL "")
SET(INSTALL_GTEST OFF CACHE INTERNAL "")
SET(gmock_build_tests OFF CACHE INTERNAL "")
SET(gtest_build_samples OFF CACHE INTERNAL "")
SET(gtest_build_tests OFF CACHE INTERNAL "")
SET(gtest_disable_pthreads OFF CACHE INTERNAL "")
SET(gtest_hide_internal_symbols OFF CACHE INTERNAL "")
SET(gtest_force_shared_crt OFF CACHE INTERNAL "")

add_subdirectory(googletest)

##############################
###        Project         ###
##############################
project(marco VERSION 1.0 LANGUAGES CXX)
enable_testing()
include("CPack")
include(CTest)

##############################
###        Globals         ###
##############################

# Set the default build type if none was specified

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'Release' as none was specified")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"  "MinSizeRel" "RelWithDebInfo")
endif()

# Disable exceptions and RTTI
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")

#set_tests_properties(${noArgsTests} PROPERTIES TIMEOUT 10)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# inherit build options from the LLVM build we are linking against
include(HandleLLVMOptions)

# SUNDIALS
set(SUNDIALS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sundials_libraries/install"
	CACHE PATH "Path to sundials_libraries/install folder")
set(SUNDIALS_INCLUDE_DIR ${SUNDIALS_DIR}/include)
set(SUNDIALS_LIBRARY_DIR ${SUNDIALS_DIR}/lib)

find_library(SUNDIALS_SOLVER_LIB sundials_ida ${SUNDIALS_LIBRARY_DIR})
find_library(SUNDIALS_NVEC_LIB sundials_nvecserial ${SUNDIALS_LIBRARY_DIR})
find_library(SUNDIALS_MANYVEC_LIB sundials_nvecmanyvector ${SUNDIALS_LIBRARY_DIR})
find_library(SUNDIALS_SUNLINSOLKLU_LIB sundials_sunlinsolklu ${SUNDIALS_LIBRARY_DIR})

add_library(sundials INTERFACE)
target_include_directories(sundials INTERFACE ${SUNDIALS_INCLUDE_DIR})
target_include_directories(sundials INTERFACE ${SUNDIALS_INCLUDE_DIRS})
target_link_libraries(sundials INTERFACE
	${SUNDIALS_SOLVER_LIB}
	${SUNDIALS_NVEC_LIB}
	${SUNDIALS_MANYVEC_LIB}
	${SUNDIALS_SUNLINSOLKLU_LIB})

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

# MARCO profiling for compilation and runtime
if(NOT DEFINED MARCO_PROFILING)
	set(MARCO_PROFILING OFF)
endif()

if (${MARCO_PROFILING} STREQUAL ON)
	add_compile_definitions(MARCO_PROFILING)
endif()


##############################
###       Subfolders       ###
##############################
include(marcoMacros.cmake)
include(extra.cmake)
add_subdirectory(tool)
add_subdirectory(lib)
add_subdirectory(test)

##############################
###          BOOST         ###
##############################
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.45.0 COMPONENTS graph exception REQUIRED)

add_library(boost INTERFACE)
target_include_directories(boost INTERFACE ${Boost_INCLUDE_DIRS})
target_compile_definitions(boost INTERFACE -DBOOST_NO_EXCEPTIONS)
target_compile_features(boost INTERFACE cxx_std_17)
