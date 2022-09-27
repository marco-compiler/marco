set(SOURCES
    AccessFunction.cpp
    Dependency.cpp
    Dumpable.cpp
    Graph.cpp
    IndexSet.cpp
    LocalMatchingSolutions.cpp
    LocalMatchingSolutionsMCIM.cpp
    LocalMatchingSolutionsVAF.cpp
    Matching.cpp
    MCIM.cpp
    MultidimensionalRange.cpp
    Point.cpp
    Range.cpp
    Scheduling.cpp)

marco_add_library(modeling
    ${SOURCES}

    LINK_COMPONENTS
    Core
    Support)
