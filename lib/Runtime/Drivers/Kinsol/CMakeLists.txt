set(LIBS
    PUBLIC argh
    PUBLIC marco::runtimeSolversKINSOL)

marco_add_runtime_library(runtimeDriverKINSOL
    STATIC
    CLI.cpp

    LINK_LIBS
    ${LIBS})
