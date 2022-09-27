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

marco_add_library(variableFilter
    ${SOURCES}

    LINK_LIBS
    marco::diagnostic
    marco::parser

    LINK_COMPONENTS
    Core
    Support)
