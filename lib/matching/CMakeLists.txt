add_library(matching SHARED src/Matching.cpp src/Edge.cpp src/Flow.cpp src/MatchingErrors.cpp src/VVarDependencyGraph.cpp) 
add_library(modelica::matching ALIAS matching )
target_include_directories(matching 
	PRIVATE 
		src 
	PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
		)
target_compile_features(matching PUBLIC cxx_std_17)
target_link_libraries(matching PUBLIC modelica::model ${Boost_LIBRARIES})

modelicaInstall(matching)
add_subdirectory(test)
