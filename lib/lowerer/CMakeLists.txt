add_library(lowerer SHARED src/Lowerer.cpp src/LowererUtils.cpp src/ExpLowerer.cpp src/CallLowerer.cpp src/OperationLowerer.cpp) 
add_library(modelica::lowerer ALIAS lowerer)
target_include_directories(lowerer 
	PRIVATE 
		src 
	PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
		)
target_compile_features(lowerer PUBLIC cxx_std_17)

llvm_map_components_to_libnames(llvm_lowerer_libs core ${LLVM_TARGETS_TO_BUILD})
target_link_libraries(lowerer PUBLIC ${llvm_lowerer_libs} modelica::utils modelica::passes)

modelicaInstall(lowerer)

add_subdirectory(test)
