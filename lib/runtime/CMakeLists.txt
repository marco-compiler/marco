set(SOURCES
		BuiltInFunctions.cpp
		MemoryManagement.cpp
		Profiling.cpp
		Runtime.cpp
		UtilityFunctions.cpp)

marco_add_library(runtime SHARED
		${SOURCES}

		LINK_COMPONENTS
		Support)
