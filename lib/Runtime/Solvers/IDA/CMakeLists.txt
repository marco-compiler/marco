set(LIBS
    PUBLIC SUNDIALS::KLU
    PUBLIC SUNDIALS::ida
    PUBLIC SUNDIALS::kinsol
    PUBLIC SUNDIALS::nvecserial
    PUBLIC SUNDIALS::sunlinsolklu)

marco_add_runtime_library(runtimeSolversIDA
    STATIC
    Instance.cpp
    Options.cpp
    Profiler.cpp

    LINK_LIBS
    ${LIBS})
