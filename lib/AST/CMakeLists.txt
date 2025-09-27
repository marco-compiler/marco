set(SOURCES
    ASTNode.cpp)

marco_add_library(AST
    ${SOURCES}

    LINK_LIBS

    LINK_COMPONENTS
    Core
    Support)

add_subdirectory(BaseModelica)
