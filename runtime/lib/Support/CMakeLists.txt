set(SOURCES
        BuiltInFunctions.cpp
        Math.cpp
        MemoryManagement.cpp
        Print.cpp
        UtilityFunctions.cpp)

marco_add_runtime_static_library(Support
        STATIC
        ${SOURCES})
