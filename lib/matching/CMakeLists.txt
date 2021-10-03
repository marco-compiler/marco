set(SOURCES
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

marco_add_library(matching SHARED ${SOURCES})

llvm_config(matching USE_SHARED core support)

target_link_libraries(matching
		PUBLIC marco::model
		PUBLIC boost
		PRIVATE pthread)

add_subdirectory(test)
