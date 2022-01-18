set(SOURCES
    AccessFunction.cpp
    Dependency.cpp
    Dumpable.cpp
    Graph.cpp
    LocalMatchingSolutions.cpp
    Matching.cpp
    MCIM.cpp
    MCIS.cpp
    MultidimensionalRange.cpp
    Point.cpp
    Range.cpp
    Scheduling.cpp)

marco_add_library(modeling SHARED
    ${SOURCES}

    LINK_LIBS
    PRIVATE marco::utils

    LINK_COMPONENTS
    Core
    Support)
