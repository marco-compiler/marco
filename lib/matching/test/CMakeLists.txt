set(SOURCES
        src/DataStructuresTest.cpp
        src/GraphTest.cpp
        src/RealScenarios.cpp)

marco_add_unittest(MARCOTest MatchingTest ${SOURCES})

marco_link_llvm_libs(MatchingTest core support)

target_link_libraries(MatchingTest
        PUBLIC marco::matching)
