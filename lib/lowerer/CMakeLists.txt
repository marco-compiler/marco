marcoAddLibrary(
	lowerer SHARED 
	src/Lowerer.cpp
   	src/LowererUtils.cpp
   	src/ExpLowerer.cpp
   	src/CallLowerer.cpp
   	src/OperationLowerer.cpp)

marco_link_llvm_libs(lowerer core bitwriter)

target_link_libraries(
	lowerer PUBLIC 
   	marco::utils
   	marco::passes)

