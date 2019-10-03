ADD_LIBRARY(lowerer SHARED src/SimExp.cpp src/Simulation.cpp) 
ADD_LIBRARY(modelica::lowerer ALIAS lowerer)
TARGET_INCLUDE_DIRECTORIES(lowerer 
	PRIVATE 
		src 
	PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
		)
TARGET_COMPILE_FEATURES(lowerer PUBLIC cxx_std_17)

llvm_map_components_to_libnames(llvm_lowerer_libs core Support ${LLVM_TARGETS_TO_BUILD})
TARGET_LINK_LIBRARIES(lowerer PUBLIC ${llvm_lowerer_libs} modelica::utils)

include(GNUInstallDirs)
INSTALL(TARGETS lowerer EXPORT lowererTargets
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION bin
	INCLUDES DESTINATION include)

INSTALL(EXPORT lowererTargets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}
	FILE lowererTargets.cmake
	NAMESPACE modelica::)

include(CMakePackageConfigHelpers)
write_basic_package_version_file("lowererConfigVersion.cmake"
	VERSION ${example_VERSION}
	COMPATIBILITY SameMajorVersion)

INSTALL(FILES lowererConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/lowererConfigVersion.cmake
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(test)
