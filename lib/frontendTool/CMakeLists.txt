set(SOURCES
    ExecuteCompilerInvocation.cpp)

marco_add_library(frontendTool SHARED
    ${SOURCES}

    LINK_LIBS
    marco::frontend

    LINK_COMPONENTS
    Core
    Support)
