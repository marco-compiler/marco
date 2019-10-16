add_library(solver SHARED src/SolvedSet.cpp src/ModelJacobian.cpp src/Model.cpp src/SimEqu.cpp src/SimVar.cpp src/DependencyGraph.cpp) 
add_library(modelica::solver ALIAS solver)
target_include_directories(solver 
	PRIVATE 
		src 
	PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
		)
target_compile_features(solver PUBLIC cxx_std_17)
target_link_libraries(solver PUBLIC modelica::simulation)

modelicaInstall(solver)
add_subdirectory(test)
