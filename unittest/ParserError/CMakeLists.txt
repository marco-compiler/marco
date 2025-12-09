set(SOURCES
    EditDistanceTest.cpp)

set(LLVM_LINK_COMPONENTS
    Core
    Support)

marco_add_unittest(ParserErrorTest ${SOURCES})

target_link_libraries(ParserErrorTest
    PRIVATE marco::Distance)
