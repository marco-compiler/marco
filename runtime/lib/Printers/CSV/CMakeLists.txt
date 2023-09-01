marco_add_runtime_static_library(PrinterCSV
    STATIC
    ../Printer.cpp
    CLI.cpp
    Options.cpp
    Printer.cpp
    Profiler.cpp)

target_link_libraries(PrinterCSV
    PUBLIC
    CLI
    Profiling)
