set(SOURCES
    src/Diagnostic.cpp
    src/Location.cpp
    src/LogMessage.cpp)

marco_add_library_new(diagnostic SHARED
    ${SOURCES}

    LINK_LIBS
    marco::utils

    LINK_COMPONENTS
    Option
    Support)