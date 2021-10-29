set(SOURCES
        src/AccessFunction.cpp
        src/Graph.cpp
        src/IncidenceMatrix.cpp
        src/LocalMatchingSolutions.cpp
        src/Matching.cpp
        src/MCIM.cpp
        src/MCIS.cpp
        src/Range.cpp)

marco_add_library(matching SHARED ${SOURCES})

marco_link_llvm_libs(matching USE_SHARED core support)

target_link_libraries(matching
        PUBLIC boost)

add_subdirectory(test)
