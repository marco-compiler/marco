set(SOURCES
        src/DirectedGraphTest.cpp
        src/MatchingCommon.cpp
        src/FlatMCIMTest.cpp
        src/UndirectedGraphTest.cpp
        src/MatchingGraphTest.cpp
        src/MatchingTest.cpp
        src/MultidimensionalRangeTest.cpp
        src/MCISTest.cpp
        src/PointTest.cpp
        src/RangeTest.cpp
        src/RegularMCIMTest.cpp
        src/SimplifyTest.cpp
        src/SolveLocalMatchingProblemTest.cpp
        src/MatchingTestCases.cpp
        src/VVarDependencyGraphTest.cpp
        src/SCCTestCases.cpp)

marco_add_unittest(MARCOTest MatchingTest ${SOURCES})

marco_link_llvm_libs(MatchingTest core support)

target_link_libraries(MatchingTest
        PUBLIC marco::matching)
