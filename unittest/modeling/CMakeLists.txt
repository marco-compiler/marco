set(SOURCES
    AccessFunctionTest.cpp
    DirectedGraphTest.cpp
    FlatMCIMTest.cpp
    MatchingCommon.cpp
    MatchingGraphTest.cpp
    MatchingTest.cpp
    MatchingTestCases.cpp
    MCISTest.cpp
    MultidimensionalRangeTest.cpp
    PointTest.cpp
    RangeTest.cpp
    RegularMCIMTest.cpp
    SchedulingTest.cpp
    SimplifyTest.cpp
    SolveLocalMatchingProblemTest.cpp
    UndirectedGraphTest.cpp
    VVarDependencyGraphTest.cpp)

marco_add_unittest(ModelingTest ${SOURCES})

marco_link_llvm_libs(ModelingTest Core Support)

target_link_libraries(ModelingTest
    PRIVATE marco::modeling)
