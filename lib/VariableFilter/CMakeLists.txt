set(SOURCES
    AST.cpp
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
    marco::Lexer
    clangBasic
    clangCodeGen

    LINK_COMPONENTS
    Core
    Support)
