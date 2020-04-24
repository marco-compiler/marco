modelicaAddLibrary(
	passes SHARED 
	src/PassManager.cpp
   	src/SolveDerivatives.cpp) 

target_link_libraries(passes PUBLIC modelica::matching)
