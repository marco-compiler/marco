set(SOURCES
        src/GraphTest.cpp
        src/IncidenceMatrixTest.cpp
        src/MatchingGraphTest.cpp
        src/RangeTest.cpp
        src/RealScenarios.cpp
        src/SolveLocalMatchingProblemTest.cpp)

marco_add_unittest(MARCOTest MatchingTest ${SOURCES})

marco_link_llvm_libs(MatchingTest core support)

target_link_libraries(MatchingTest
        PUBLIC marco::matching)
