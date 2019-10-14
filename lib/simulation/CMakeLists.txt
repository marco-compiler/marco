add_library(simulation SHARED src/SimExp.cpp src/SimErrors.cpp src/SimType.cpp src/SimCall.cpp src/SimLexerStateMachine.cpp src/SimParser.cpp) 
add_library(modelica::simulation ALIAS simulation)
target_include_directories(simulation 
	PRIVATE 
		src 
	PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
		)
target_compile_features(simulation PUBLIC cxx_std_17)

llvm_map_components_to_libnames(llvm_lowerer_libs Support)
target_link_libraries(simulation PUBLIC ${llvm_lowerer_libs} modelica::utils)

modelicaInstall(simulation)

add_subdirectory(test)
