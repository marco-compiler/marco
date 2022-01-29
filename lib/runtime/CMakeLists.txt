set(SOURCES
		BuiltInFunctions.cpp
		Math.cpp
		MemoryManagement.cpp
		Profiling.cpp
		Runtime.cpp
		UtilityFunctions.cpp)

# The runtime library is forced to be built as a static one.
# This is needed because it will leverage the simulation functions ('init', 'step', etc.) that are generated
# from the compiler. It is not possible to compile a shared library with such extern functions.
# Moreover, apart from this technical detail, it makes more sense to just provide one single executable
# rather than the both the executable and the shared runtime library.

marco_add_library(runtime STATIC
		${SOURCES}

		LINK_COMPONENTS
		Support)
