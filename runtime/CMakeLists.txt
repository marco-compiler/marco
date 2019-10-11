ADD_LIBRARY(runtime STATIC src/PrintFunctions.cpp) 
ADD_LIBRARY(modelica::runtime ALIAS runtime)
TARGET_INCLUDE_DIRECTORIES(runtime 
	PRIVATE 
		src 
	PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
		)
TARGET_COMPILE_FEATURES(runtime PUBLIC cxx_std_17)


llvm_map_components_to_libnames(llvm_runtime_libs core Support ${LLVM_TARGETS_TO_BUILD})
TARGET_LINK_LIBRARIES(runtime PUBLIC ${llvm_runtime_libs} modelica::utils)

include(GNUInstallDirs)
INSTALL(TARGETS runtime EXPORT runtimeTargets
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION bin
	INCLUDES DESTINATION include)

INSTALL(EXPORT runtimeTargets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}
	FILE runtimeTargets.cmake
	NAMESPACE modelica::)

include(CMakePackageConfigHelpers)
write_basic_package_version_file("runtimeConfigVersion.cmake"
	VERSION ${example_VERSION}
	COMPATIBILITY SameMajorVersion)

INSTALL(FILES runtimeConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/runtimeConfigVersion.cmake
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(test)
