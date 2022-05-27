set(SOURCES
    AST.cpp
    Error.cpp
    Filter.cpp
    LexerStateMachine.cpp
    Parser.cpp
    Range.cpp
    Token.cpp
    Tracker.cpp
    VariableFilter.cpp)

marco_add_library(variableFilter SHARED
    ${SOURCES}

    DEPENDS
    marco::utils

    LINK_LIBS
    marco::diagnostic
    marco::utils

    LINK_COMPONENTS
    Core
    Support)
