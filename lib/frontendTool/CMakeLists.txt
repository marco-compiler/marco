set(SOURCES
    ExecuteCompilerInvocation.cpp)

marco_add_library(frontendTool SHARED
    ${SOURCES}

    DEPENDS
    marco::frontend

    LINK_LIBS
    PRIVATE marco::frontend

    LINK_COMPONENTS
    Core
    Support)
