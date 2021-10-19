set(SOURCES
		src/MatchingTest.cpp
		src/VVarDependencyGraphTest.cpp
		src/SccCollapsingTest.cpp)

marco_add_unittest(MARCOTest MatchingTest ${SOURCES})

marco_link_llvm_libs(MatchingTest core support)

target_link_libraries(MatchingTest
		PRIVATE marco::matching)
