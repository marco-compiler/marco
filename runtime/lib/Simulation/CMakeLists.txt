# Declare the library driving the simulation.
marco_add_runtime_static_library(Simulation
    STATIC
    Profiler.cpp
    Runtime.cpp)

target_link_libraries(Simulation
    PUBLIC
    CLI
    Profiling)

# Declare the library used to make the simulation executable.
marco_add_runtime_static_library(Starter
    STATIC
    Starter.cpp)

target_link_libraries(Starter PUBLIC Simulation)
