add_library(omcToModel SHARED src/OmcToModelPass.cpp) 
add_library(modelica::omcToModel ALIAS omcToModel)
target_include_directories(omcToModel 
	PRIVATE 
		src 
	PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
		)
target_compile_features(omcToModel PUBLIC cxx_std_17)
target_link_libraries(omcToModel PUBLIC modelica::frontend)

modelicaInstall(omcToModel)
add_subdirectory(test)
