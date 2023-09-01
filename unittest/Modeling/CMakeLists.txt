set(SOURCES
    AccessFunctionTest.cpp
    AccessFunctionRotoTranslationTest.cpp
    Common.cpp
    CyclesTest.cpp
    DirectedGraphTest.cpp
    IndexSetListTest.cpp
    IndexSetRTreeTest.cpp
    MCIMTest.cpp
    MultidimensionalRangeTest.cpp
    PointTest.cpp
    RangeTest.cpp
    SolveLocalMatchingProblemTest.cpp
    UndirectedGraphTest.cpp)

marco_add_unittest(ModelingTest ${SOURCES})

marco_link_llvm_libs(ModelingTest Core Support)

target_link_libraries(ModelingTest
    PRIVATE marco::modeling)
