set(LIBS
    PUBLIC argh)

marco_add_runtime_library(runtimeSupport
    STATIC
    BuiltInFunctions.cpp
    CLI.cpp
    Math.cpp
    MemoryManagement.cpp
    Options.cpp
    Print.cpp
    UtilityFunctions.cpp

    LINK_LIBS
    ${LIBS})
