set(LIBS
    PUBLIC argh
    PUBLIC marco::runtimeSolversIDA)

marco_add_runtime_library(runtimeDriverIDA
    STATIC
    ../Driver.cpp
    CLI.cpp
    Driver.cpp

    LINK_LIBS
    ${LIBS})
