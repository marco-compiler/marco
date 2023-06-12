set(SOURCES
    BuiltInFunctions.cpp
    Math.cpp
    MemoryManagement.cpp
    Print.cpp
    UtilityFunctions.cpp)

marco_add_runtime_library(runtimeSupport
    STATIC
    ${SOURCES})

marco_add_runtime_library(runtimeSupportShared
    SHARED
    PARTIAL_SOURCES_INTENDED
    ${SOURCES})

if (MARCO_PROFILING)
    target_compile_definitions(runtimeSupport PUBLIC MARCO_PROFILING)
endif()
