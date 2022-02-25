set(SOURCES
    Diagnostic.cpp
    Location.cpp
    LogMessage.cpp)

if(MINGW)
    set(LINK_MODE STATIC)
else()
    set(LINK_MODE SHARED)
endif()
marco_add_library(diagnostic ${LINK_MODE}
    ${SOURCES}

    LINK_LIBS
    PRIVATE marco::utils

    LINK_COMPONENTS
    Option
    Support)
