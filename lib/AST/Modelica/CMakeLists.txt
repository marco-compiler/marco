set(SOURCES
    Class.cpp
    Equation.cpp
    Expression.cpp
    Modification.cpp
    Node.cpp)

marco_add_library(ASTModelica
    ${SOURCES}

    LINK_LIBS
    PUBLIC
    marco::AST
    marco::Lexer

    LINK_COMPONENTS
    Core
    Support)
