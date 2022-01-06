set(SOURCES
        AccessFunction.cpp
        Dumpable.cpp
        Graph.cpp
        LocalMatchingSolutions.cpp
        Matching.cpp
        MCIM.cpp
        MCIS.cpp
        MultidimensionalRange.cpp
        Point.cpp
        Range.cpp)

marco_add_library(modeling SHARED
    ${SOURCES}

    LINK_LIBS
    marco::utils

    LINK_COMPONENTS
    Core
    Support)
