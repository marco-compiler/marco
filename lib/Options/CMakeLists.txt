set(SOURCES
    Options.cpp)

marco_add_library(options
    ${SOURCES}

    DEPENDS
    MARCOOptions

    LINK_COMPONENTS
    Option
    Support)
