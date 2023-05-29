set(LIBS
    PUBLIC argh)

marco_add_runtime_library(runtimeCLI
    STATIC
    Category.cpp
    CLI.cpp

    LINK_LIBS
    ${LIBS})
