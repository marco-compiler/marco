set(LIBS
    PUBLIC argh
    PUBLIC marco::runtimeSolversEulerForward)

marco_add_runtime_library(runtimeDriverEulerForward
    STATIC
    ../Driver.cpp
    CLI.cpp
    Driver.cpp

    LINK_LIBS
    ${LIBS})

if (MARCO_PROFILING)
    target_compile_definitions(runtimeDriverEulerForward PUBLIC MARCO_PROFILING)
endif()
