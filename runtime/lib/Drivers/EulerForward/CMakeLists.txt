marco_add_runtime_static_library(DriverEulerForward
    STATIC
    ../Driver.cpp
    CLI.cpp
    Driver.cpp)

target_link_libraries(DriverEulerForward
    PUBLIC
    CLI
    SolverEulerForward)
