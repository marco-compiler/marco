set(SOURCES
		src/BuiltInFunctions.cpp
		src/MemoryManagement.cpp
		src/OldRuntime.cpp
		src/Profiling.cpp
		src/Runtime.cpp
		src/UtilityFunctions.cpp)

marco_add_library(runtime SHARED ${SOURCES})

marco_link_llvm_libs(runtime USE_SHARED support)

add_subdirectory(test)
