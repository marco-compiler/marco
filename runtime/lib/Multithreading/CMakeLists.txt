marco_add_runtime_static_library(Multithreading
    STATIC
    CLI.cpp
    Options.cpp
    ThreadPool.cpp)

target_link_libraries(Multithreading
    PUBLIC
    CLI)

if (Threads_FOUND)
    target_link_libraries(Multithreading
        PUBLIC
        Threads::Threads)
endif()
