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

set(default_build_type "Debug")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (CMAKE_BUILD_TYPE MATCHES Debug)
	message(STATUS "Enabling abi breaking checks.")
	SET(LLVM_ENABLE_ABI_BREAKING_CHECKS ON "Enable this if you are using a llvm built in debug mode, turn it off for release")
else(CMAKE_BUILD_TYPE MATCHES Debug)
	message(STATUS "Disabling abi breaking checks.")
	SET(LLVM_ENABLE_ABI_BREAKING_CHECKS OFF "Enable this if you are using a llvm built in debug mode, turn it off for release")
endif(CMAKE_BUILD_TYPE MATCHES Debug)
add_subdirectory(googletest)

##############################
###        Project         ###
##############################

project(modelica VERSION 1.0 LANGUAGES CXX)
enable_testing()
include("CPack")
include(CTest)
set(CMAKE_DEBUG_POSTFIX "-d")

##############################
###        Globals         ###
##############################
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti -Wall -Werror -Wextra -Wno-unused-variable -Wno-unused-parameter")
set_tests_properties(${noArgsTests}   PROPERTIES TIMEOUT 10)
find_package(LLVM REQUIRED 9.0.0)
link_directories(${LLVM_PATH}/lib)
include_directories(${LLVM_PATH}/include)

if (LLVM_ENABLE_ABI_BREAKING_CHECKS)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLLVM_ENABLE_ABI_BREAKING_CHECKS=1")
endif(LLVM_ENABLE_ABI_BREAKING_CHECKS)

##############################
###       InstallMacro     ###
##############################
macro(modelicaInstall target)

include(GNUInstallDirs)
INSTALL(TARGETS ${target} EXPORT ${target}Targets
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION bin
	INCLUDES DESTINATION include)

INSTALL(EXPORT ${target}Targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}
	FILE ${target}Targets.cmake
	NAMESPACE modelica::)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(${target}ConfigVersion.cmake
	VERSION ${example_VERSION}
	COMPATIBILITY SameMajorVersion)

INSTALL(FILES ${target}Config.cmake ${CMAKE_CURRENT_BINARY_DIR}/${target}ConfigVersion.cmake
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

endmacro(modelicaInstall)

##############################
###       Subfolders       ###
##############################
add_subdirectory(frontend)
add_subdirectory(lowerer)
add_subdirectory(utils)
add_subdirectory(tool)
add_subdirectory(runtime)
add_subdirectory(simulationGenerator)

##############################
###        DOXYGEN         ###
##############################
find_package(Doxygen)
if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    # note the option ALL which allows to build the docs together with the application
    add_custom_target(doc 
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)

##############################
###        COVERAGE        ###
##############################
if (${CMAKE_CXX_FLAGS} MATCHES ".*--coverage.*")
	find_program(LCOV NAMES lcov)
	find_program(GENHTML NAMES genhtml)

	add_custom_command(
		OUTPUT base.info
		COMMAND ${LCOV} --capture --initial --directory . --output-file ./base.info
		VERBATIM
		)

	add_custom_command(
		OUTPUT filtered.info 
		COMMAND ${LCOV} --remove ./total.info "/usr/*" "*/googletest/*" "*/build/*" --output-file ./filtered.info
		DEPENDS total.info
		VERBATIM)

	add_custom_command(
		OUTPUT total.info 
		COMMAND ${LCOV} --add-tracefile ./base.info --add-tracefile ./test.info --output-file ./total.info
		DEPENDS test.info base.info
		VERBATIM)

	add_custom_command(
		OUTPUT test.info 
		COMMAND ${LCOV} --capture --directory . --output-file ./test.info
		VERBATIM)

	add_custom_target(genCoverage
		COMMAND ${GENHTML} ./filtered.info --output-directory ./Coverage
		DEPENDS filtered.info
		BYPRODUCTS ./Coverage
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Generate coverage"
		VERBATIM )

	add_custom_target(coverage COMMAND rm test.info filtered.info base.info total.info)
	add_dependencies(coverage genCoverage)
	add_dependencies(coverage test)
elseif(TRUE)
	add_custom_target(coverage
		COMMAND echo could not calculate coverage run cmake with -DCMAKE_CXX_FLAGS=--coverage
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Coverage not aviable"
		VERBATIM)
endif()

