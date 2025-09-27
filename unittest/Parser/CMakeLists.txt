set(SOURCES
    FloatLexerTest.cpp
    IntegerLexerTest.cpp
    LexerTest.cpp
    ParserTest.cpp)

marco_add_unittest(ParserTest ${SOURCES})

marco_link_llvm_libs(ParserTest Core Support)

target_link_libraries(ParserTest
    PRIVATE marco::ParserBaseModelica)
