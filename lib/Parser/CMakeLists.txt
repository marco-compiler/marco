set(SOURCES
    Message.cpp
    ModelicaStateMachine.cpp
    Parser.cpp
    Token.cpp)

marco_add_library(parser SHARED
    ${SOURCES}

    LINK_LIBS
    PUBLIC marco::diagnostic
    PUBLIC marco::ast

    LINK_COMPONENTS
    Core
    Support)
