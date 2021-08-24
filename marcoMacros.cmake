##############################
###       InstallMacro     ###
##############################
macro(marcoInstall target)

include(GNUInstallDirs)
INSTALL(TARGETS ${target} EXPORT ${target}Targets
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION bin
	INCLUDES DESTINATION include)

INSTALL(EXPORT ${target}Targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}
	FILE ${target}Targets.cmake
	NAMESPACE marco::)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(${target}ConfigVersion.cmake
	VERSION ${example_VERSION}
	COMPATIBILITY SameMajorVersion)

INSTALL(FILES ${target}Config.cmake ${CMAKE_CURRENT_BINARY_DIR}/${target}ConfigVersion.cmake
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

endmacro(marcoInstall)

##############################
###    addLibraryMacro     ###
##############################
macro(marcoAddLibrary target)
	add_library(${target} ${ARGN})

add_library(marco::${target} ALIAS ${target})

target_include_directories(${target}
	PRIVATE 
		src 
	PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
		)

target_compile_features(${target} PUBLIC cxx_std_17)
marcoInstall(${target})

add_subdirectory(test)
	

endmacro(marcoAddLibrary)


##############################
### marcoAddTestMacro   ###
##############################
macro(marcoAddTest target)
	include(GoogleTest)

	add_executable(${target}Test ${ARGN})
	add_executable(marco::${target}Test ALIAS ${target}Test) 
	target_link_libraries(${target}Test PRIVATE gtest gtest_main ${target})
	target_include_directories(${target}Test PUBLIC include PRIVATE src)
	target_compile_features(${target}Test PUBLIC cxx_std_17)

	gtest_add_tests(TARGET     ${target}Test
					TEST_SUFFIX .noArgs
					TEST_LIST   noArgsTests
	)

endmacro(marcoAddTest)

##############################
###  marcoAddToolMacro  ###
##############################
macro(marcoAddTool target)

	add_executable(${target} src/Main.cpp)
	add_executable(marco::${target} ALIAS ${target})

	target_link_libraries(${target} PUBLIC ${ARGN})
	target_compile_features(${target} PUBLIC cxx_std_17)

	include(GNUInstallDirs)
	INSTALL(TARGETS ${target} RUNTIME DESTINATION bin)

	add_subdirectory(test)
	

endMacro(marcoAddTool)


###
###  marco_link_llvm_libs  
###
###  Same as llvm_config, but automatically turns on shared libs if the
###  LLVM we are linking with has them available. Everything is linked as
###  PUBLIC so that everything works with the way we use CMake.
###    "Shouldn't have this been done by LLVM's Cmake stuff already?" You bet!
###  "Does it do that then?" NO!!
###
function(marco_link_llvm_libs target)
  set(link_components ${ARGN})

	if(LLVM IN_LIST LLVM_AVAILABLE_LIBS)
	  if (DEFINED link_components AND DEFINED LLVM_DYLIB_COMPONENTS)
      if("${LLVM_DYLIB_COMPONENTS}" STREQUAL "all")
        set(link_components "")
      else()
        list(REMOVE_ITEM link_components ${LLVM_DYLIB_COMPONENTS})
      endif()
    endif()

    target_link_libraries(${target} PUBLIC LLVM)
	endif()
	
	llvm_map_components_to_libnames(libs ${link_components})
	target_link_libraries(${target} PUBLIC ${libs})
endfunction(marco_link_llvm_libs)

