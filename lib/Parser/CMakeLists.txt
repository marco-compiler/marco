set(SOURCES
        BaseModelica/Parser.cpp
        BaseModelica/StateMachine.cpp
        BaseModelica/Token.cpp
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
