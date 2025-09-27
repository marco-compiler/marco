set(SOURCES
    Parser.cpp
    StateMachine.cpp
    Token.cpp)

marco_add_library(ParserBaseModelica
    ${SOURCES}

    LINK_LIBS
    PUBLIC
    marco::Lexer
    marco::ASTBaseModelica

    LINK_COMPONENTS
    Core
    Support

    LINK_LIBS PUBLIC
    MLIRIR

    clangBasic
    clangCodeGen)
