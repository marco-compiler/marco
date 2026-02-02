set(SOURCES
    AccessFunction.cpp
    AccessFunctionRotoTranslation.cpp
    Common.cpp
    Cycles.cpp
    DirectedGraph.cpp
    IndexSet.cpp
    MCIM.cpp
    MultidimensionalRange.cpp
    Point.cpp
    Range.cpp
    RTree.cpp
    SolveLocalMatchingProblem.cpp
    UndirectedGraph.cpp
    GraphDumper.cpp
    GraphDumperMermaidBackend.cpp
)

set(LLVM_LINK_COMPONENTS
    Core
    Support)

marco_add_unittest(ModelingTest ${SOURCES})

target_link_libraries(ModelingTest
    PRIVATE marco::modeling)
