set(SOURCES
    AccessFunctionTest.cpp
    AccessFunctionRotoTranslationTest.cpp
    Common.cpp
    CyclesTest.cpp
    DirectedGraphTest.cpp
    IndexSetTest.cpp
    MCIMTest.cpp
    MultidimensionalRange.cpp
    Point.cpp
    Range.cpp
    RTree.cpp
    SolveLocalMatchingProblemTest.cpp
    UndirectedGraphTest.cpp
    GraphDumperTest.cpp
    GraphDumperMermaidBackendTest.cpp
)

set(LLVM_LINK_COMPONENTS
    Core
    Support)

marco_add_unittest(ModelingTest ${SOURCES})

target_link_libraries(ModelingTest
    PRIVATE marco::modeling)
