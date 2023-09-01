set(SOURCES
    AccessFunction.cpp
    AccessFunctionConstant.cpp
    AccessFunctionEmpty.cpp
    AccessFunctionRotoTranslation.cpp
    AccessFunctionZeroDims.cpp
    AccessFunctionZeroResults.cpp
    Dependency.cpp
    Dumpable.cpp
    Graph.cpp
    IndexSet.cpp
    IndexSetList.cpp
    IndexSetRTree.cpp
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

    LINK_LIBS PUBLIC
    MLIRIR

    LINK_COMPONENTS
    Core
    Support)
