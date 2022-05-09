set(SOURCES
    Diagnostic.cpp
    Location.cpp
    LogMessage.cpp)

marco_add_library(diagnostic ${LINK_MODE}
    ${SOURCES}

    LINK_LIBS
    PRIVATE marco::utils

    LINK_COMPONENTS
    Option
    Support)
