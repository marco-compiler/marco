# Find the thread library of the system.
find_package(Threads REQUIRED)

set(LIBS
    PUBLIC ${CMAKE_THREAD_LIBS_INIT}
    PUBLIC argh)

marco_add_runtime_library(runtimeMultithreading
    STATIC
    CLI.cpp
    Options.cpp
    ThreadPool.cpp

    LINK_LIBS
    ${LIBS})
