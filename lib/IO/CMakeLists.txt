set(SOURCES
    Command.cpp
    InputFile.cpp)

marco_add_library(IO
    ${SOURCES}

    LINK_COMPONENTS
    Support)
