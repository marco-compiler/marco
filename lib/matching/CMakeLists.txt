add_library(matching SHARED src/Matching.cpp src/Edge.cpp src/Flow.cpp src/MatchingErrors.cpp src/VVarDependencyGraph.cpp src/SVarDependencyGraph.cpp src/SCCDependencyGraph.cpp src/Schedule.cpp) 
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
target_link_libraries(matching PUBLIC modelica::model )
target_compile_definitions(matching PUBLIC -DBOOST_NO_EXCEPTIONS)

modelicaInstall(matching)
add_subdirectory(test)
