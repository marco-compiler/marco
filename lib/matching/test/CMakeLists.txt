set(SOURCES
        src/Common.cpp
        src/FlatMCIMTest.cpp
        src/GraphTest.cpp
        src/MatchingGraphTest.cpp
        src/MatchingTest.cpp
        src/MultidimensionalRangeTest.cpp
        src/MCISTest.cpp
        src/PointTest.cpp
        src/RangeTest.cpp
        src/RegularMCIMTest.cpp
        src/SimplifyTest.cpp
        src/SolveLocalMatchingProblemTest.cpp
        src/TestCases.cpp)

marco_add_unittest(MARCOTest MatchingTest ${SOURCES})

marco_link_llvm_libs(MatchingTest core support)

target_link_libraries(MatchingTest
        PUBLIC marco::matching)
