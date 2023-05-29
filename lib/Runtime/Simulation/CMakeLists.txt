set(LIBS
    PUBLIC argh)

marco_add_runtime_library(runtime
    STATIC
    Profiler.cpp
    Runtime.cpp

    LINK_LIBS
    ${LIBS})
