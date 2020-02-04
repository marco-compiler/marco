add_library(model SHARED src/ModExp.cpp src/ModErrors.cpp src/Assigment.cpp src/ModType.cpp src/ModCall.cpp src/ModLexerStateMachine.cpp src/ModParser.cpp src/SolvedSet.cpp src/ModelJacobian.cpp src/Model.cpp src/ModEquation.cpp src/ModVariable.cpp src/DependencyGraph.cpp src/VectorAccess.cpp src/ModConst.cpp)
add_library(modelica::model ALIAS model)
target_include_directories(model 
	PRIVATE 
		src 
	PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
		)
target_compile_features(model PUBLIC cxx_std_17)

target_link_libraries(model PUBLIC  modelica::utils)

modelicaInstall(model)

add_subdirectory(test)
