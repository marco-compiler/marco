set(SOURCES
    DynamicDimensionsGraph.cpp)

marco_add_library(ASTAnalysis
    ${SOURCES}

    LINK_LIBS
    PUBLIC marco::ast

    LINK_COMPONENTS
    Core
    Support)
