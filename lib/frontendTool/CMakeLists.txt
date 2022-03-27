set(SOURCES
    ExecuteCompilerInvocation.cpp)

if(MINGW OR MSVC)
    set(LINK_MODE STATIC)
else()
    set(LINK_MODE SHARED)
endif()
marco_add_library(frontendTool ${LINK_MODE}
    ${SOURCES}

    DEPENDS
    marco::frontend

    LINK_LIBS
    PRIVATE marco::frontend

    LINK_COMPONENTS
    Core
    Support)
