set(SHARED_SOURCES
	BuiltInFunctions.cpp
    IDA.cpp
	Print.cpp
	Math.cpp
	MemoryManagement.cpp
	Profiling.cpp
	UtilityFunctions.cpp)

set(STATIC_SOURCES
	${SHARED_SOURCES}
	CLI.cpp
	Runtime.cpp)

# Fetch the Argh! project, which is used to handle runtime command-line arguments

FetchContent_Declare(
	argh
	URL https://github.com/adishavit/argh/archive/refs/tags/v1.3.2.zip
)

FetchContent_MakeAvailable(argh)

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

# Libraries to be linked
set(LIBS
	PUBLIC SUNDIALS::KLU
	PUBLIC SUNDIALS::ida
	PUBLIC SUNDIALS::nvecserial
	PUBLIC SUNDIALS::sunlinsolklu)

# Declare the shared library
marco_add_runtime_library(runtimeShared
	SHARED
	PARTIAL_SOURCES_INTENDED
	${SHARED_SOURCES}

	LINK_LIBS
	${LIBS})

# Declare the static library
marco_add_runtime_library(runtime
	STATIC
	${STATIC_SOURCES}

	LINK_LIBS PUBLIC
	${LIBS}
	argh)

# Enable the CLI only for the static runtime library
target_compile_definitions(runtime PUBLIC MARCO_CLI)
