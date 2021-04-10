modelicaAddLibrary(
	passes SHARED 
	src/PassManager.cpp
	src/SolveModel.cpp
	src/ForwardEuler.cpp)

target_link_libraries(passes PUBLIC modelica::matching)
