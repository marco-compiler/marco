set(SOURCES
    LexerTest.cpp
    ParserTest.cpp)

marco_add_unittest(ParserTest ${SOURCES})

marco_link_llvm_libs(ParserTest Core Support)

target_link_libraries(ParserTest
    PUBLIC marco::parser)
