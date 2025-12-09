set(SOURCES
    FloatLexerTest.cpp
    IntegerLexerTest.cpp
    LexerTest.cpp
    ParserTest.cpp)

set(LLVM_LINK_COMPONENTS
    Core
    Support)

marco_add_unittest(ParserTest ${SOURCES})

target_link_libraries(ParserTest
    PRIVATE marco::ParserBaseModelica)
