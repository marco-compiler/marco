modelicaAddLibrary(
	matching SHARED 
	src/Matching.cpp
   	src/Edge.cpp
   	src/Flow.cpp
   	src/MatchingErrors.cpp
   	src/VVarDependencyGraph.cpp
   	src/SVarDependencyGraph.cpp
   	src/SCCDependencyGraph.cpp
   	src/Schedule.cpp
	src/SccLookup.cpp
	src/SccCollapsing.cpp) 

target_link_libraries(
	matching PUBLIC 
	modelica::model 
	pthread
	graph)

