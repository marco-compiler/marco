modelicaAddLibrary(
	passes SHARED 
	src/PassManager.cpp
        src/SolveModel.cpp)

target_link_libraries(passes PUBLIC modelica::matching)
