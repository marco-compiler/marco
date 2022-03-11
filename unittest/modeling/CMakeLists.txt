set(SOURCES
    AccessFunctionTest.cpp
    CyclesTest.cpp
    DirectedGraphTest.cpp
    MatchingCommon.cpp
    MatchingGraphTest.cpp
    MatchingTest.cpp
    MatchingTestCases.cpp
    MCIMFlatTest.cpp
    MCIMRegularTest.cpp
    MCISTest.cpp
    MultidimensionalRangeTest.cpp
    MultidimensionalRangeRaggedTest.cpp
    PointTest.cpp
    RangeTest.cpp
    SchedulingTest.cpp
    SimplifyTest.cpp
    SolveLocalMatchingProblemTest.cpp
    UndirectedGraphTest.cpp)

marco_add_unittest(ModelingTest ${SOURCES})

marco_link_llvm_libs(ModelingTest Core Support)

target_link_libraries(ModelingTest
    PRIVATE marco::modeling  PRIVATE marco::utils)
