ADD_LIBRARY(runtime STATIC src/PrintFunctions.cpp) 
ADD_LIBRARY(modelica::runtime ALIAS runtime)
TARGET_INCLUDE_DIRECTORIES(runtime 
	PRIVATE 
		src 
	PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
		)
TARGET_COMPILE_FEATURES(runtime PUBLIC cxx_std_17)


llvm_map_components_to_libnames(llvm_runtime_libs core Support ${LLVM_TARGETS_TO_BUILD})
TARGET_LINK_LIBRARIES(runtime PUBLIC ${llvm_runtime_libs} modelica::utils)

modelicaInstall(runtime)

add_subdirectory(test)
