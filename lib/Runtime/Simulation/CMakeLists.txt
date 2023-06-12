set(LIBS
    PUBLIC argh)

marco_add_runtime_library(runtime
    STATIC
    Profiler.cpp
    Runtime.cpp

    LINK_LIBS
    ${LIBS})

if (MARCO_PROFILING)
    target_compile_definitions(runtime PUBLIC MARCO_PROFILING)
endif()
