modelicaAddLibrary(
	lowerer SHARED 
	src/Lowerer.cpp
   	src/LowererUtils.cpp
   	src/ExpLowerer.cpp
   	src/CallLowerer.cpp
   	src/OperationLowerer.cpp)

llvm_map_components_to_libnames(llvm_lowerer_libs
   	core
   	${LLVM_TARGETS_TO_BUILD})

target_link_libraries(
	lowerer PUBLIC 
	${llvm_lowerer_libs}
   	modelica::utils
   	modelica::passes)

