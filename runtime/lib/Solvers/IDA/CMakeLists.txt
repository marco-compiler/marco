marco_add_runtime_static_library(SolverIDA
    STATIC
    Instance.cpp
    Options.cpp
    Profiler.cpp)

if (MARCO_ENABLE_SUNDIALS)
    target_link_libraries(SolverIDA
        PUBLIC
        SUNDIALS::KLU
        SUNDIALS::ida
        SUNDIALS::kinsol
        SUNDIALS::nvecserial
        SUNDIALS::sunlinsolklu)
endif()

target_link_libraries(SolverIDA
    PUBLIC
    SolverSUNDIALS
    Profiling)
