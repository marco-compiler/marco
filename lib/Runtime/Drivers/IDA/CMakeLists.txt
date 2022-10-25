set(LIBS
    PUBLIC marco::runtime
    PUBLIC argh)

marco_add_runtime_library(runtimeDriverIDA
    STATIC
    ../Driver.cpp
    CLI.cpp
    Driver.cpp

    LINK_LIBS
    ${LIBS})

if (MARCO_PROFILING)
    target_compile_definitions(runtimeDriverIDA PUBLIC MARCO_PROFILING)
endif()
