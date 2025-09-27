set(SOURCES
    ModelicaStateMachine.cpp
    Parser.cpp
    Token.cpp
    Location.cpp)

marco_add_library(parser
    ${SOURCES}

    LINK_LIBS
    PUBLIC marco::ASTBaseModelica

    LINK_COMPONENTS
    Core
    Support

    LINK_LIBS PUBLIC
    MLIRIR

    clangBasic
    clangCodeGen)
# TODO check if really need the link libs
