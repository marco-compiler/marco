set(SOURCES
    FloatLexer.cpp
    IntegerLexer.cpp)

set(LLVM_LINK_COMPONENTS
    Core
    Support)

marco_add_unittest(LexerTest ${SOURCES})

target_link_libraries(LexerTest
    PRIVATE marco::Lexer)
