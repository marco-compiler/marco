set(SOURCES
    EditDistanceTest.cpp)

marco_add_unittest(ParserErrorTest ${SOURCES})

target_link_libraries(ParserErrorTest
    PRIVATE marco::Distance)
