marco_add_runtime_static_library(DriverKINSOL
    STATIC
    CLI.cpp)

target_link_libraries(DriverKINSOL
    PUBLIC
    CLI
    SolverKINSOL)
