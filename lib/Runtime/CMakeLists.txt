# Fetch the Argh! project, which is used to handle runtime command-line
# arguments.

FetchContent_Declare(
	argh
	URL https://github.com/adishavit/argh/archive/refs/tags/v1.3.2.zip)

FetchContent_MakeAvailable(argh)

# We build two versions of the runtime library: one is shared and the other is
# static.
# The shared library is used only for JIT tests, which needs to rely on a
# shared library for the runtime functions.
# The static library is the one that will link to generate the real simulation.
# Differently from the shared library, it also contains the simulation driver
# (the 'runSimulation' function) and leverages the simulation functions
# ('init', 'step', etc.) that are generated from the compiler while converting
# the model. It is not possible to compile a shared library with such extern
# functions. Moreover, apart from this technical detail, it makes more sense to
# just provide one single executable rather than the both the executable and
# the shared runtime library.

add_custom_target(marco-runtime)

add_subdirectory(CLI)
add_subdirectory(Profiling)
add_subdirectory(Multithreading)
add_subdirectory(Modeling)
add_subdirectory(Support)
add_subdirectory(Printers)
add_subdirectory(Solvers)
add_subdirectory(Drivers)
add_subdirectory(Simulation)
