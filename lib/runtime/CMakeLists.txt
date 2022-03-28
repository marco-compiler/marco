set(SHARED_SOURCES
		d2fixed.cpp
		Nostdlib.cpp
		Fdlibm.cpp
		BuiltInFunctions.cpp
		Math.cpp
		MemoryManagement.cpp
		Profiling.cpp
		UtilityFunctions.cpp)

set(STATIC_SOURCES
		${SHARED_SOURCES}
		Runtime.cpp)

# We build two versions of the runtime library: one is shared and the other is static.
# The shared libraries is used only for JIT tests, which needs to rely on a shared library
# for the runtime functions.
# The static library is the one that will link to generate the real simulation. Differently
# from the shared library, it also contains the simulation driver (the 'runSimulation'
# function) and leverages the simulation functions ('init', 'step', etc.) that are generated
# from the compiler while converting the model. It is not possible to compile a shared
# library with such extern functions. Moreover, apart from this technical detail, it makes
# more sense to just provide one single executable rather than the both the executable and
# the shared runtime library.

add_custom_target(marco-runtime)

# Declare the shared library
marco_add_library(runtimeShared
		SHARED
		PARTIAL_SOURCES_INTENDED
		${SHARED_SOURCES})

if(MSVC)
target_compile_options(runtimeShared PRIVATE "/Oi-")
target_link_options(runtimeShared PRIVATE "/NODEFAULTLIB")
elseif(MINGW)
# -fno-strict-aliasing needed for fdlibm
target_compile_options(runtimeShared PRIVATE "-fno-strict-aliasing")
target_link_options(runtimeShared PRIVATE "-nostdlib")
endif()

add_dependencies(marco-runtime marco::runtimeShared)

# Declare the static library
marco_add_library(runtime
		STATIC
		${STATIC_SOURCES})

if(MSVC)
target_compile_options(runtime PRIVATE "/Oi-")
target_link_options(runtime PRIVATE "/NODEFAULTLIB")
elseif(MINGW)
# -fno-strict-aliasing needed for fdlibm
target_compile_options(runtime PRIVATE "-fno-strict-aliasing" PRIVATE "-g" PRIVATE "-O0" PRIVATE "/Oi-")
target_link_options(runtime PRIVATE "-nostdlib")
endif()

add_dependencies(marco-runtime marco::runtime)

