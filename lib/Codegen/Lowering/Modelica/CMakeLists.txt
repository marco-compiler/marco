set(SOURCES
    ClassTree.cpp
    Equations.cpp
    Expressions.cpp
    Lowerer.cpp
    Types.cpp)

marco_add_library(LoweringModelica
    ${SOURCES}

    LINK_LIBS PUBLIC
    marco::ASTModelica
    marco::Lexer
    MLIRModelica)
