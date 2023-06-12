marco_add_runtime_library(runtimeProfiling
    STATIC
    Profiler.cpp
    Profiling.cpp
    Statistics.cpp
    Timer.cpp)

if (MARCO_PROFILING)
    target_compile_definitions(runtimeProfiling PUBLIC MARCO_PROFILING)
endif()
