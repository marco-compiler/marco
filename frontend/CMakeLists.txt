ADD_LIBRARY(frontend SHARED src/Lexer.cpp src/LexerStateMachine.cpp)
ADD_LIBRARY(modelica::frontend ALIAS frontend)
TARGET_INCLUDE_DIRECTORIES(frontend 
	PRIVATE 
		src 
	PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
		)
TARGET_COMPILE_FEATURES(frontend PUBLIC cxx_std_17)

llvm_map_components_to_libnames(llvm_frontend_libs core Support ${LLVM_TARGETS_TO_BUILD})
TARGET_LINK_LIBRARIES(frontend PUBLIC ${llvm_frontend_libs})

include(GNUInstallDirs)
INSTALL(TARGETS frontend EXPORT frontendTargets
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION bin
	INCLUDES DESTINATION include)

INSTALL(EXPORT frontendTargets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}
	FILE frontendTargets.cmake
	NAMESPACE modelica::)

include(CMakePackageConfigHelpers)
write_basic_package_version_file("frontendConfigVersion.cmake"
	VERSION ${example_VERSION}
	COMPATIBILITY SameMajorVersion)

INSTALL(FILES frontendConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/frontendConfigVersion.cmake
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

ADD_SUBDIRECTORY(test)
