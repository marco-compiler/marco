add_library(passes SHARED src/PassManager.cpp src/SolveDerivatives.cpp src/Matching.cpp) 
add_library(modelica::passes ALIAS passes)
target_include_directories(passes 
	PRIVATE 
		src 
	PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
		)
target_compile_features(passes PUBLIC cxx_std_17)
target_link_libraries(passes PUBLIC modelica::model)

modelicaInstall(passes)
add_subdirectory(test)
