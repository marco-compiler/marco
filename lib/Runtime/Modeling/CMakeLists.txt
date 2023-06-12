marco_add_runtime_library(runtimeModeling
    STATIC
    IndexSet.cpp
    MultidimensionalRange.cpp
    Range.cpp)

if (MARCO_PROFILING)
    target_compile_definitions(runtimeModeling PUBLIC MARCO_PROFILING)
endif()
