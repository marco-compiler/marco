set(SOURCES
    Lexer.cpp
    Parser.cpp)

set(LLVM_LINK_COMPONENTS
    Core
    Support)

marco_add_unittest(ParserModelicaTest ${SOURCES})

target_link_libraries(ParserModelicaTest
    PRIVATE marco::ParserModelica)
