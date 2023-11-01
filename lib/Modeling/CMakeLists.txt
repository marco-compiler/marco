set(SOURCES
    AccessFunction.cpp
    AccessFunctionConstant.cpp
    AccessFunctionEmpty.cpp
    AccessFunctionRotoTranslation.cpp
    Dependency.cpp
    DimensionAccess.cpp
    DimensionAccessAdd.cpp
    DimensionAccessConstant.cpp
    DimensionAccessDimension.cpp
    DimensionAccessDiv.cpp
    DimensionAccessIndices.cpp
    DimensionAccessMul.cpp
    DimensionAccessSub.cpp
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
