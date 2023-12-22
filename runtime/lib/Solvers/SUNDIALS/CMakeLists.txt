marco_add_runtime_static_library(SolverSUNDIALS
    STATIC
    Instance.cpp
    Options.cpp
    Profiler.cpp)

if (MARCO_ENABLE_SUNDIALS)
    target_link_libraries(SolverSUNDIALS
        PUBLIC
        SUNDIALS::KLU
        SUNDIALS::nvecserial)
endif()

target_link_libraries(SolverSUNDIALS
    PUBLIC
    Profiling)
