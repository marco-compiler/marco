set(SOURCES
    Lexer.cpp
    Parser.cpp
    Token.cpp)

marco_add_library(ParserModelica
    ${SOURCES}

    LINK_LIBS
    PUBLIC
    marco::Lexer
    marco::ASTModelica

    LINK_COMPONENTS
    Core
    Support

    LINK_LIBS PUBLIC
    clangBasic)
