marco_add_runtime_static_library(SolverKINSOL
    STATIC
    Instance.cpp
    Options.cpp
    Profiler.cpp)

if (MARCO_ENABLE_SUNDIALS)
    target_link_libraries(SolverKINSOL
        PUBLIC
        SUNDIALS::KLU
        SUNDIALS::kinsol
        SUNDIALS::nvecserial
        SUNDIALS::sunlinsolklu)
endif()
