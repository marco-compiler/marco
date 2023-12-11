# Fetch the Argh! project, which is used to handle command-line arguments.
include(FetchContent)

FetchContent_Declare(
    argh
    GIT_REPOSITORY https://github.com/adishavit/argh.git
    GIT_TAG v1.3.2)

FetchContent_MakeAvailable(argh)

install(TARGETS argh
        EXPORT MARCORuntimeTargets)

# Declare the library used to make the simulation executable.
marco_add_runtime_static_library(CLI
    STATIC
    Category.cpp
    CLI.cpp)

target_link_libraries(CLI PUBLIC argh)
