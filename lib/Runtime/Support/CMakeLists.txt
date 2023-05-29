set(LIBS
    PUBLIC argh
    PUBLIC marco::runtimeProfiling)

marco_add_runtime_library(runtimeSupport
    STATIC
    BuiltInFunctions.cpp
    CLI.cpp
    Math.cpp
    MemoryManagement.cpp
    Options.cpp
    Print.cpp
    Profiler.cpp
    UtilityFunctions.cpp

    LINK_LIBS
    ${LIBS})
