modelicaAddLibrary(
	runtime SHARED 
	src/PrintFunctions.cpp) 

llvm_map_components_to_libnames(llvm_runtime_libs core Support ${LLVM_TARGETS_TO_BUILD})
TARGET_LINK_LIBRARIES(runtime PUBLIC ${llvm_runtime_libs} modelica::utils)


