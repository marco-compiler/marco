set(LIBS
    PUBLIC SUNDIALS::KLU
    PUBLIC SUNDIALS::kinsol
    PUBLIC SUNDIALS::nvecserial
    PUBLIC SUNDIALS::sunlinsolklu)

marco_add_runtime_library(runtimeSolversKINSOL
    STATIC
    Instance.cpp
    Options.cpp
    Profiler.cpp

    LINK_LIBS
    ${LIBS})

if (MARCO_PROFILING)
    target_compile_definitions(runtimeSolversKINSOL PUBLIC MARCO_PROFILING)
endif()
