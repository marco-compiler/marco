set(LIBS
    PUBLIC marco::runtime
    PUBLIC argh)

marco_add_runtime_library(runtimeDriverKINSOL
    STATIC
    CLI.cpp

    LINK_LIBS
    ${LIBS})

if (MARCO_PROFILING)
    target_compile_definitions(runtimeDriverKINSOL PUBLIC MARCO_PROFILING)
endif()
