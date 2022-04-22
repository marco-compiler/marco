set(SOURCES
    VariableFilter.cpp)

marco_add_library(variableFilter SHARED
    ${SOURCES}

    DEPENDS
    marco::utils

    LINK_LIBS
    marco::utils

    LINK_COMPONENTS
    Core
    Support)
