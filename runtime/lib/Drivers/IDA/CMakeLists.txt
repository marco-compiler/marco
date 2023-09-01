marco_add_runtime_static_library(DriverIDA
    STATIC
    ../Driver.cpp
    CLI.cpp
    Driver.cpp)

target_link_libraries(DriverIDA
    PUBLIC
    CLI
    SolverIDA)
