set(SOURCES
    src/AccessFunctionTest.cpp
    src/DirectedGraphTest.cpp
    src/FlatMCIMTest.cpp
    src/MatchingCommon.cpp
    src/MatchingGraphTest.cpp
    src/MatchingTest.cpp
    src/MatchingTestCases.cpp
    src/MCISTest.cpp
    src/MultidimensionalRangeTest.cpp
    src/PointTest.cpp
    src/RangeTest.cpp
    src/RegularMCIMTest.cpp
    src/SimplifyTest.cpp
    src/SolveLocalMatchingProblemTest.cpp
    src/UndirectedGraphTest.cpp
    src/VVarDependencyGraphTest.cpp)

marco_add_unittest(MARCOTest ModelingTest ${SOURCES})

marco_link_llvm_libs(ModelingTest core support)

target_link_libraries(ModelingTest
    PUBLIC marco::modeling)
