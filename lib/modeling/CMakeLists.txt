set(SOURCES
        src/AccessFunction.cpp
        src/Dumpable.cpp
        src/Graph.cpp
        src/LocalMatchingSolutions.cpp
        src/Matching.cpp
        src/MCIM.cpp
        src/MCIS.cpp
        src/MultidimensionalRange.cpp
        src/Point.cpp
        src/Range.cpp)

marco_add_library(modeling SHARED ${SOURCES})

marco_link_llvm_libs(modeling USE_SHARED core support)

target_link_libraries(modeling
        PUBLIC marco::utils)

add_subdirectory(test)
