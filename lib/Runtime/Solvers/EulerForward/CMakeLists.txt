marco_add_runtime_library(runtimeSolversEulerForward
    STATIC
    Options.cpp
    Profiler.cpp)

if (MARCO_PROFILING)
    target_compile_definitions(runtimeSolversEulerForward PUBLIC MARCO_PROFILING)
endif()
