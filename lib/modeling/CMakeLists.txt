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
    MCIMFlat.cpp
    MCIMRegular.cpp
    MultidimensionalRange.cpp
    Point.cpp
    Range.cpp
    Scheduling.cpp)

if(MINGW OR MSVC)
    set(LINK_MODE STATIC)
else()
    set(LINK_MODE SHARED)
endif()
marco_add_library(modeling ${LINK_MODE}
    ${SOURCES}

    LINK_LIBS
    PRIVATE marco::utils

    LINK_COMPONENTS
    Core
    Support)
