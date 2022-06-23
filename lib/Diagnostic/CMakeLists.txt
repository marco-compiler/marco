set(SOURCES
    Diagnostic.cpp
    Location.cpp
    LogMessage.cpp
    Printer.cpp)

marco_add_library(diagnostic
    ${SOURCES}

    LINK_LIBS
    PRIVATE marco::utils

    LINK_COMPONENTS
    Option
    Support)
