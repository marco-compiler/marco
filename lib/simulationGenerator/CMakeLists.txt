add_library(simulationGenerator STATIC src/Generator.cpp)
add_library(modelica::simulationGenerator  ALIAS simulationGenerator) 
target_link_libraries(simulationGenerator PUBLIC modelica::lowerer)
target_include_directories(simulationGenerator PRIVATE src 
	PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
	)
target_compile_features(simulationGenerator PUBLIC cxx_std_17)

modelicaInstall(simulationGenerator)

