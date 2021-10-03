set(SOURCES
		src/MatchingTest.cpp
		src/VVarDependencyGraphTest.cpp
		src/SccCollapsingTest.cpp)

marco_add_unittest(MARCOTest MatchingTest ${SOURCES})

llvm_config(MatchingTest core support)

target_link_libraries(MatchingTest
		PRIVATE marco::matching)
