marco_add_runtime_static_library(SolverEulerForward
    STATIC
    Options.cpp
    Profiler.cpp)

target_link_libraries(SolverEulerForward
    PUBLIC
    Profiling)
